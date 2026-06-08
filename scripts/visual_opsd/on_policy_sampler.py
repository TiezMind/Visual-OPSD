"""On-policy completion sampling for Visual-OPSD training.

During training, the student must generate its own completion (reasoning +
answer) from the current policy, and the teacher then *scores* that very
completion under a privileged prompt.  This module implements the first
half of that loop: given the raw student inputs (problem image + question),
produce a list of sampled token ids.

Key design points
-----------------

1. The underlying model is an FSDP-wrapped ``Bagel``.  To run the BAGEL
   inference primitives (``prepare_prompts``, ``forward_cache_update_text``,
   ``prepare_vit_images``, ``forward_cache_update_vit``, ``generate_text``)
   we need a module whose parameters are materialised on every rank.  We
   use ``FSDP.summon_full_params(..., recurse=True, writeback=False)``
   which gathers every nested FSDP unit for the duration of the context.

2. BAGEL's ``generate_text`` supports batch size one only.  The training
   loop is expected to call ``generate`` once per raw sample.  Gradient
   accumulation and data parallelism happen at the outer level.

3. The returned token ids strip BOTH the leading BOS (inserted by
   ``prepare_start_tokens``) AND the trailing EOS (the break condition
   inside ``generate_text`` does not append eos).  The training pack
   builder wraps the completion with its own ``<|im_start|>``/``<|im_end|>``
   pair via ``pack_sequence``, so we do not add them here.

4. BAGEL's inference primitives internally invoke FSDP-wrapped submodules
   (ViT transformer, Qwen2 decoder layers).  Each such call goes through
   ``FullyShardedDataParallel.forward``, whose post-forward logic
   **reshards** the flat parameter — even when we are inside
   ``summon_full_params(recurse=True)``.  That re-shard violates the
   exit invariant of ``summon_full_params`` ("expects tensor to be
   unsharded" assertion).  We therefore bypass ``FSDP.forward`` on every
   FSDP instance for the duration of sampling (see
   ``_bypass_fsdp_forward``).  Parameters are guaranteed unsharded by
   the enclosing summon context, so the pass-through to
   ``_fsdp_wrapped_module`` is safe.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._flat_param import FlatParamHandle

from modeling.bagel.qwen2_navit import NaiveCache


@dataclass
class SamplingConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = True
    system_prompt: Optional[str] = None
    instruction_suffix: Optional[str] = None  # appended to question text
    # Because the UMM student is heavily fine-tuned on interleaved CoT data,
    # an overfit student often emits ``<image_start>`` then EOS in
    # anticipation of image generation; a single text-only ``generate_text``
    # round therefore stops with only ``<think>...</think><image_start>`` as
    # the completion.  To keep the on-policy completion well-formed text we
    # inject a ``<|vision_end|>`` sentinel after each premature
    # ``<image_start>`` and keep sampling.  This is the maximum number of
    # such image-skip rounds to attempt (0 disables the fix and produces
    # legacy one-shot behaviour).
    max_image_skips: int = 2


# ------------------------------------------------------------------ #
# FSDP precision patch
# ------------------------------------------------------------------ #


@contextlib.contextmanager
def _patch_fsdp_force_full_precision_off() -> Iterator[None]:
    """Temporarily make ``FlatParamHandle._force_full_precision`` always False.

    Why
    ---
    Inside ``FSDP.summon_full_params`` the handle sets its training state to
    ``SUMMON_FULL_PARAMS``.  That in turn makes ``_force_full_precision``
    evaluate to ``True`` whenever mixed-precision training is enabled, so
    FSDP allocates the unsharded ``_full_prec_full_param_padded`` tensor
    in FP32 rather than reusing the BF16 ``_full_param_padded``.  For a
    ~14.5 B-parameter model that is the difference between a 29 GB and a
    58 GB per-rank allocation, which easily OOMs an 80 GB GPU once you
    account for:

      * the sharded FP32 master weights of the student (~7 GB),
      * the sharded FP32 master weights of the EMA teacher (~7 GB),
      * the AdamW state (~14 GB), and
      * the CUDA context + activations.

    What
    ----
    We override the ``_force_full_precision`` property at the class level
    to return ``False``.  This diverts FSDP onto the normal mixed-precision
    path:

      * ``pre_unshard`` -> ``_use_low_precision_shard`` (allocate
        ``_mp_shard`` in BF16, downcast from the FP32 master),
      * ``_get_padded_unsharded_flat_param`` returns the BF16
        ``_full_param_padded``,
      * ``_all_gather_flat_param`` targets the BF16 tensor,
      * ``post_unshard`` frees ``_mp_shard``.

    The FP32 master shards (``_local_shard``) are never touched, so on
    context exit all optimizer state remains intact.

    Safety
    ------
    * We restore the original property on exit (success or exception).
    * The override affects every ``FlatParamHandle`` instance globally for
      the duration of the context; this is acceptable because sampling is
      synchronous and there is only one FSDP tree in the training process.
    * Only valid under ``torch.no_grad`` (we never flow gradients through
      the BF16 views).
    """
    cls = FlatParamHandle
    sentinel = object()
    original = cls.__dict__.get("_force_full_precision", sentinel)
    cls._force_full_precision = property(lambda self: False)
    try:
        yield
    finally:
        if original is sentinel:
            try:
                delattr(cls, "_force_full_precision")
            except AttributeError:
                pass
        else:
            setattr(cls, "_force_full_precision", original)


# ------------------------------------------------------------------ #
# FSDP bypass context
# ------------------------------------------------------------------ #


@contextlib.contextmanager
def _bypass_fsdp_forward(root_module: nn.Module) -> Iterator[None]:
    """Temporarily override ``forward`` on every ``FullyShardedDataParallel``
    instance in ``root_module`` so it is a direct pass-through to the
    wrapped module.

    Rationale
    ---------
    In wrapper-mode FSDP (what we use), the pre/post-forward unshard and
    reshard logic is invoked *inline* inside
    ``FullyShardedDataParallel.forward`` — not via ``register_forward_hook``.
    That means the only way to prevent an inner FSDP module from
    resharding its flat parameter when it is called is to avoid running
    ``FSDP.forward`` at all.  While we are inside
    ``summon_full_params(recurse=True)`` all flat parameters are already
    unsharded, so a direct call to the wrapped module will read the
    correct full-precision tensors and produce the right output.

    Safety
    ------
    * Only valid inside ``summon_full_params(recurse=True, writeback=False)``.
    * We operate under ``torch.no_grad`` in the caller, so the
      post-forward backward-hook registration we skip is unnecessary.
    * ``_root_pre_forward`` one-time setup has already been performed
      during normal training forwards, and no new allocation is needed
      here.
    """

    def _direct_forward(self: FSDP, *args: Any, **kwargs: Any) -> Any:
        return self._fsdp_wrapped_module(*args, **kwargs)

    patched: List[FSDP] = []
    for mod in root_module.modules():
        if isinstance(mod, FSDP):
            mod.__dict__["forward"] = MethodType(_direct_forward, mod)
            patched.append(mod)
    try:
        yield
    finally:
        for mod in patched:
            mod.__dict__.pop("forward", None)


class OnPolicySampler:
    """Generate text completions from the *current* FSDP student model."""

    def __init__(
        self,
        tokenizer,
        vit_transform,
        new_token_ids: Dict[str, int],
        cfg: SamplingConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_device(generation_input: Dict[str, Any], device) -> Dict[str, Any]:
        out = {}
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _build_student_prompt_text(self, raw: Dict[str, Any]) -> str:
        """Compose the question-only text that precedes the completion.

        We intentionally avoid including the reference reasoning here —
        only the student gets this prompt.  Matches ``build_student_sample``
        in ``data/opsd_pack_builder.py``.
        """
        text = raw["question_text"]
        if self.cfg.instruction_suffix:
            text = f"{text}{self.cfg.instruction_suffix}"
        return text

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        fsdp_model,
        raw: Dict[str, Any],
        device,
        dtype: torch.dtype = torch.bfloat16,
        rank0_only: bool = False,
    ) -> List[int]:
        """Sample one completion under the student prompt.

        Returns the list of token ids (excluding the leading BOS and any
        trailing EOS — ready to be wrapped by the pack builder).

        Parameters
        ----------
        rank0_only:
            If True, materialize the full unsharded student on global rank 0
            only and broadcast the sampled completion ids to every other
            rank.  All ranks still enter the summon context (collective
            all-gather), but ``_full_param_padded`` is allocated on rank 0
            only, saving ~one full-model BF16 copy of GPU memory
            (~29 GB for a 14.5 B MoT backbone) on every non-rank-0 rank —
            which is the dominant transient allocation that puts
            ``summon_full_params`` over the 80 GB/GPU budget once FP32
            AdamW state and a forward-only fixed_teacher are resident.
            Because the student is identical across ranks and RNG is
            identical within a data-parallel group, there is no loss of
            sampling diversity relative to sampling-on-every-rank.
        """
        eos_id = int(self.new_token_ids["eos_token_id"])
        bos_id = int(self.new_token_ids["bos_token_id"])

        dist_ready = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if dist_ready else 1
        my_rank = dist.get_rank() if dist_ready else 0
        # Only flip the rank0-only path on when we actually have more than
        # one rank — otherwise there is nothing to save and nothing to
        # broadcast.
        use_rank0_only = bool(rank0_only) and world_size > 1
        is_sampler_rank = (not use_rank0_only) or my_rank == 0

        # `summon_full_params` gathers *all* FSDP-owned parameters on every
        # rank.  ``writeback=False`` asks FSDP not to copy-back any
        # modifications — we're read-only here.  ``recurse=True`` is the
        # default and what we need given BAGEL's nested FSDP layout.
        #
        # ``_patch_fsdp_force_full_precision_off`` forces the unsharded
        # tensors to stay in BF16 instead of being upcast to FP32 (which
        # would double the per-rank memory footprint and OOM on 80 GB GPUs
        # for the ~14.5 B parameter MoT backbone).
        #
        # ``_bypass_fsdp_forward`` replaces every FSDP instance's
        # ``forward`` with a direct pass-through so that BAGEL's inference
        # primitives, which invoke nested FSDP submodules via their
        # ``__call__``, do NOT accidentally reshard params mid-generation
        # (which would break the summon_full_params exit invariant).
        #
        # Release any cached but unused allocator blocks before we ask
        # FSDP to allocate the full unsharded BF16 parameter tensor — on
        # 80 GB H800s with three FSDP models and FP32 AdamW state the
        # allocation is within a few hundred MiB of the budget, and a
        # single fragmented segment is enough to tip it into OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with _patch_fsdp_force_full_precision_off(), FSDP.summon_full_params(
            fsdp_model,
            recurse=True,
            writeback=False,
            offload_to_cpu=False,
            rank0_only=use_rank0_only,
        ), _bypass_fsdp_forward(fsdp_model):
            if is_sampler_rank:
                # FSDP module.__getattr__ forwards to the inner Bagel for
                # any missing attribute.  All the methods we call below
                # are @torch.no_grad inside BAGEL, so gradient state is
                # neutral.
                model = fsdp_model
                was_training = model.training
                model.eval()
                try:
                    with torch.autocast(
                        device_type="cuda",
                        enabled=torch.cuda.is_available(),
                        dtype=dtype,
                    ):
                        generated_ids = self._generate_single(
                            model=model,
                            raw=raw,
                            device=device,
                            eos_id=eos_id,
                        )
                finally:
                    if was_training:
                        model.train()
            else:
                generated_ids = None

        # generated_ids is shape [T, 1] (BAGEL's generate_text).  Convert
        # and strip the leading BOS that prepare_start_tokens inserts.
        if is_sampler_rank:
            if isinstance(generated_ids, torch.Tensor):
                token_list = generated_ids.view(-1).tolist()
            else:
                token_list = [int(t) for t in generated_ids] if generated_ids is not None else []
            if token_list and int(token_list[0]) == bos_id:
                token_list = token_list[1:]
            while token_list and int(token_list[-1]) == eos_id:
                token_list = token_list[:-1]
            token_list = [int(t) for t in token_list]
        else:
            token_list = None

        # Broadcast completion ids from rank 0 to all other ranks in the
        # rank0_only path.  We broadcast *after* exiting summon so that
        # the extra buffer is reclaimed on rank 0 before the collective
        # runs (minor, but avoids holding the ~29 GB unshard buffer
        # across the broadcast).
        if use_rank0_only:
            token_list = broadcast_completion(token_list, src_rank=0)

        return token_list if token_list is not None else []

    # ------------------------------------------------------------------ #
    # internal: build prompt KV cache and sample
    # ------------------------------------------------------------------ #

    def _generate_single(
        self,
        model,
        raw: Dict[str, Any],
        device,
        eos_id: int,
    ) -> torch.Tensor:
        """Run BAGEL inference primitives; return [T, 1] token tensor.

        When the policy is overfit on interleaved CoT, a single
        ``generate_text`` round typically stops right after the model
        emits ``<image_start>`` (followed by ``<|im_end|>``), giving a
        degenerate completion such as ``<think>..</think><image_start>``
        with no reasoning tail and no answer.  Training on that induces
        a "length-collapse" failure mode.

        To avoid this, after detecting ``<image_start>`` in the round's
        decoded text we close the current chat turn by appending a lone
        ``<|im_end|>`` to the cache (``generate_text`` broke before
        forwarding the sampled EOS, so the cache is missing it), and
        let the student keep sampling.  The second round then naturally
        starts with ``<image_end>`` — which is how the training data's
        second block begins — followed by the post-image reasoning and
        ``<answer>..</answer>``.  We concatenate the rounds directly
        (dropping each round's leading BOS) to obtain the full
        interleaved chain.  Up to ``max_image_skips`` such skips are
        performed; if the last round still ends on ``<image_start>`` we
        stop and return what we have.

        Note that ``<image_start>`` and ``<image_end>`` are *literal*
        multi-token strings ([27, 1805, 4906, 29] / [27, 1805, 6213, 29])
        in the Qwen tokenizer — NOT the ``<|vision_start|>/<|vision_end|>``
        special ids.  So we detect via ``tokenizer.decode`` + substring
        instead of comparing token ids.
        """
        cache = NaiveCache(model.config.llm_config.num_hidden_layers)
        kv_lens = [0]
        ropes = [0]

        bos_id = int(self.new_token_ids["bos_token_id"])
        max_skips = max(0, int(getattr(self.cfg, "max_image_skips", 0)))

        # 1) Optional system prompt.
        if self.cfg.system_prompt:
            generation_input, kv_lens, ropes = model.prepare_prompts(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                prompts=[self.cfg.system_prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._to_device(generation_input, device)
            cache = model.forward_cache_update_text(cache, **generation_input)

        # 2) Problem image (ViT).
        # ``prepare_vit_images`` expects a PIL image which it transforms via
        # the passed ``transforms``.  Our raw envelope already has a
        # transformed tensor, but passing it back out as an image would be
        # wasteful — so we forward the tensor directly by subclassing the
        # call below and rebuilding the primitive inline.
        image_tensor = raw["problem_image_tensor"].to(device)
        cache, kv_lens, ropes = _forward_cache_update_vit_from_tensor(
            model=model,
            past_key_values=cache,
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_tensor=image_tensor,
            new_token_ids=self.new_token_ids,
            device=device,
        )

        # 3) Question (+ optional instruction suffix).
        prompt_text = self._build_student_prompt_text(raw)
        generation_input, kv_lens, ropes = model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[prompt_text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._to_device(generation_input, device)
        cache = model.forward_cache_update_text(cache, **generation_input)

        # 4) Iterative sampling with optional image-skip continuation.
        #
        # ``self.cfg.max_new_tokens`` is interpreted as the TOTAL budget
        # across all skip-image rounds (plus the one-token ``<|im_end|>``
        # sentinel we inject between rounds).  Historically this was a
        # per-round cap, which meant ``max_new_tokens=512`` with
        # ``max_image_skips=1`` could produce completions of up to
        # ~1024 tokens — surprising for callers that set the cap for
        # memory budgeting.  The tracked accounting includes the leading
        # BOS of each round, so with two rounds the maximum user-visible
        # completion length is ``max_new_tokens - 1 - num_rounds`` in the
        # worst case, which is close enough to the requested cap.
        total_budget = max(1, int(self.cfg.max_new_tokens))
        consumed = 0
        round_tensors: List[torch.Tensor] = []
        skips_used = 0
        while True:
            remaining = total_budget - consumed
            if remaining <= 0:
                break

            gen_input = model.prepare_start_tokens(
                kv_lens, ropes, self.new_token_ids
            )
            gen_input = self._to_device(gen_input, device)
            round_tokens = model.generate_text(
                past_key_values=cache,
                max_length=remaining,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                end_token_id=eos_id,
                **gen_input,
            )
            round_tensors.append(round_tokens)

            # ``generate_text`` forwarded every token in ``round_tokens``
            # through the cache (including the leading BOS), so both the
            # cache length and the rope offset advance by that many slots.
            round_len = int(round_tokens.shape[0])
            consumed += round_len
            kv_lens = [kv_lens[0] + round_len]
            ropes = [ropes[0] + round_len]

            if skips_used >= max_skips:
                break

            # Only retry when (a) the round actually stopped on EOS (not
            # on the remaining budget — those are already "long enough")
            # and (b) the decoded round contains ``<image_start>`` (the
            # literal ThinkMorph text marker, tokenised as multiple
            # regular tokens; see the docstring above).
            stopped_on_eos = round_len < remaining
            if not stopped_on_eos:
                break

            # No budget left for the turn-sentinel + at least one new
            # content token in the next round — return what we have.
            if total_budget - consumed <= 1:
                break

            round_list = round_tokens.view(-1).tolist()
            content_ids = (
                round_list[1:]
                if round_list and int(round_list[0]) == bos_id
                else round_list
            )
            decoded = (
                self.tokenizer.decode(content_ids) if content_ids else ""
            )
            if "<image_start>" not in decoded:
                break

            # Close the current chat turn in the cache.  ``generate_text``
            # broke right after sampling ``<|im_end|>`` without ever
            # forwarding it, so the cache still sits mid-turn.  Appending
            # the EOS lets the next ``prepare_start_tokens``-emitted
            # ``<|im_start|>`` open a fresh, properly-delimited turn.
            # We do NOT inject any ``<image_end>`` bytes here: in the
            # training data the post-image block already *starts* with
            # ``<image_end>``, so we let the policy re-emit it naturally.
            cache = _forward_cache_append_text_tokens(
                model=model,
                past_key_values=cache,
                curr_kvlen=kv_lens[0],
                curr_rope=ropes[0],
                token_ids=[eos_id],
                device=device,
            )
            kv_lens = [kv_lens[0] + 1]
            ropes = [ropes[0] + 1]
            consumed += 1  # the injected ``<|im_end|>`` counts too.

            skips_used += 1

        # 5) Flatten rounds into one ``[T, 1]`` tensor expected by the
        # caller: keep the first round's leading BOS (the outer
        # ``generate()`` strips one BOS) and drop the leading BOS of
        # every subsequent round.  Each round's content is concatenated
        # directly — the next round will already start with
        # ``<image_end>`` (training's block-2 prefix) so no extra
        # separator is needed.  The resulting completion ids look like:
        #
        #     <think>..</think><image_start><image_end><think>..
        #                                   </think><answer>..</answer>
        #
        # which is the pure-text view of the original interleaved data
        # once the VT image block is removed — exactly what Visual-OPSD with
        # ``--visual_gen False`` expects to score.
        if len(round_tensors) == 1:
            return round_tensors[0]

        first = round_tensors[0]
        pieces: List[torch.Tensor] = [first]
        for nxt in round_tensors[1:]:
            if nxt.numel() > 0 and int(nxt[0, 0].item()) == bos_id:
                nxt = nxt[1:]
            if nxt.numel() > 0:
                pieces.append(nxt)
        return torch.cat(pieces, dim=0)


# --------------------------------------------------------------------- #
# Helper: forward an already-transformed ViT tensor through the cache
# --------------------------------------------------------------------- #


def _forward_cache_update_vit_from_tensor(
    model,
    past_key_values: NaiveCache,
    curr_kvlens: List[int],
    curr_rope: List[int],
    image_tensor: torch.Tensor,
    new_token_ids: Dict[str, int],
    device,
):
    """Mirror of ``Bagel.prepare_vit_images`` + ``forward_cache_update_vit``,
    but takes a pre-transformed tensor instead of a PIL image.

    Exists to avoid re-transforming the problem image that the dataset
    already normalised / resized for ViT.
    """
    from data.data_utils import patchify

    curr_kvlen = curr_kvlens[0]
    curr_position_id = curr_rope[0]

    packed_key_value_indexes = list(range(0, curr_kvlen))

    packed_text_ids: List[int] = [int(new_token_ids["start_of_image"])]
    packed_text_indexes: List[int] = [0]
    packed_indexes: List[int] = [curr_kvlen]

    vit_position_ids = model.get_flattened_position_ids(
        image_tensor.size(1),
        image_tensor.size(2),
        model.vit_patch_size,
        max_num_patches_per_side=model.vit_max_num_patch_per_side,
    )
    vit_tokens = patchify(image_tensor, model.vit_patch_size)
    num_img_tokens = vit_tokens.shape[0]

    # indexes within the local packed block (including start/end image tokens)
    packed_vit_token_indexes = list(range(1, 1 + num_img_tokens))
    packed_indexes.extend(range(curr_kvlen + 1, curr_kvlen + 1 + num_img_tokens))
    packed_text_ids.append(int(new_token_ids["end_of_image"]))
    packed_text_indexes.append(1 + num_img_tokens)
    packed_indexes.append(curr_kvlen + 1 + num_img_tokens)

    packed_seqlens = [num_img_tokens + 2]
    packed_position_ids = [curr_position_id] * (num_img_tokens + 2)
    new_kvlens = [curr_kvlen + num_img_tokens + 2]
    new_ropes = [curr_position_id + 1]

    generation_input = {
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
        "packed_text_indexes": torch.tensor(
            packed_text_indexes, dtype=torch.long, device=device
        ),
        "vit_token_seqlens": torch.tensor(
            [num_img_tokens], dtype=torch.int, device=device
        ),
        "packed_vit_tokens": vit_tokens.to(device),
        "packed_vit_position_ids": vit_position_ids.to(device),
        "packed_vit_token_indexes": torch.tensor(
            packed_vit_token_indexes, dtype=torch.long, device=device
        ),
        "packed_position_ids": torch.tensor(
            packed_position_ids, dtype=torch.long, device=device
        ),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
        "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
        "packed_key_value_indexes": torch.tensor(
            packed_key_value_indexes, dtype=torch.long, device=device
        ),
        "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
    }

    past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)
    return past_key_values, new_kvlens, new_ropes


# --------------------------------------------------------------------- #
# Helper: append raw tokens to the text KV cache
# --------------------------------------------------------------------- #


def _forward_cache_append_text_tokens(
    model,
    past_key_values,
    curr_kvlen: int,
    curr_rope: int,
    token_ids: List[int],
    device,
):
    """Append ``token_ids`` (as-is, no auto BOS/EOS wrapping) to the text
    KV cache.

    Equivalent to ``Bagel.prepare_prompts`` + ``Bagel.forward_cache_update_text``
    minus the automatic ``[bos, ..., eos]`` wrap, so the caller has full
    control over which structural delimiters end up in the cache.  Useful
    when we need to close a previous chat turn (by appending a single
    ``<|im_end|>``) without opening and closing a whole new turn.
    """
    n = len(token_ids)
    if n == 0:
        return past_key_values

    packed_text_ids = torch.tensor(
        [int(t) for t in token_ids], dtype=torch.long, device=device
    )
    packed_text_position_ids = torch.tensor(
        list(range(curr_rope, curr_rope + n)), dtype=torch.long, device=device
    )
    text_token_lens = torch.tensor([n], dtype=torch.int, device=device)
    packed_text_indexes = torch.tensor(
        list(range(curr_kvlen, curr_kvlen + n)), dtype=torch.long, device=device
    )
    packed_key_value_indexes = torch.tensor(
        list(range(curr_kvlen)), dtype=torch.long, device=device
    )
    key_values_lens = torch.tensor([curr_kvlen], dtype=torch.int, device=device)

    generation_input = {
        "text_token_lens": text_token_lens,
        "packed_text_ids": packed_text_ids,
        "packed_text_position_ids": packed_text_position_ids,
        "packed_text_indexes": packed_text_indexes,
        "packed_key_value_indexes": packed_key_value_indexes,
        "key_values_lens": key_values_lens,
    }
    return model.forward_cache_update_text(past_key_values, **generation_input)


# --------------------------------------------------------------------- #
# Distributed broadcast helpers
# --------------------------------------------------------------------- #


def broadcast_completion(
    completion_ids: Optional[List[int]], src_rank: int = 0
) -> List[int]:
    """Broadcast a sampled completion (python list of ints) from ``src_rank``
    to all ranks.  Used in conjunction with ``rank0_only=True`` sampling
    to cut GPU memory during FSDP.summon_full_params.
    """
    if not dist.is_available() or not dist.is_initialized():
        return completion_ids or []
    obj_list = [completion_ids if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(obj_list, src=src_rank)
    return obj_list[0] or []


def _to_cpu_picklable(obj: Any) -> Any:
    """Recursively convert CUDA tensors to CPU tensors so that ``obj``
    can be pickled and broadcast via ``dist.broadcast_object_list``.
    ``dist.broadcast_object_list`` would do a CPU roundtrip anyway, so
    eagerly releasing the source-rank's CUDA copies is harmless.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=False)
    if isinstance(obj, dict):
        return {k: _to_cpu_picklable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [_to_cpu_picklable(v) for v in obj]
        return type(obj)(conv) if isinstance(obj, tuple) else conv
    return obj


def _to_device(obj: Any, device) -> Any:
    """Inverse of ``_to_cpu_picklable``: move leaf tensors onto ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [_to_device(v, device) for v in obj]
        return type(obj)(conv) if isinstance(obj, tuple) else conv
    return obj


def broadcast_raw(
    raw: Dict[str, Any],
    src_rank: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Broadcast a ``raw`` dict (mix of tensors, strings, ints, nested
    dicts) from ``src_rank`` to all ranks.  Required when paired with
    ``rank0_only=True`` sampling: every rank must train on the SAME
    ``(raw, completion)`` pair or the student/teacher forward sees
    mismatched (question, answer) and the distillation loss becomes
    meaningless.

    This effectively reduces the per-step effective batch size by
    ``world_size`` because every rank processes the same sample; callers
    should increase ``gradient_accumulation_steps`` accordingly to keep
    the tokens-per-optimizer-step budget constant.

    Large image tensors are round-tripped through pickle (CPU) which is
    not the most efficient path, but the OPSD batch is b=1 and the
    problem image is a few MB, so the wall-clock cost is negligible
    relative to the ~O(seconds) saved vs. FSDP summon on every rank.
    """
    if not dist.is_available() or not dist.is_initialized():
        return raw
    world = dist.get_world_size()
    if world <= 1:
        return raw
    my_rank = dist.get_rank()
    if my_rank == src_rank:
        payload: List[Any] = [_to_cpu_picklable(raw)]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=src_rank)
    received = payload[0] if payload[0] is not None else {}
    if device is not None:
        received = _to_device(received, device)
    return received
