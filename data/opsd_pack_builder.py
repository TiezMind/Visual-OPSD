"""Single-sample packing helpers for Visual-OPSD on-policy training.

Given a RAW envelope from ``OPSDPairedIterableDataset`` and a student-sampled
completion (``completion_ids``), these helpers build *two* packed batches
compatible with ``Bagel.forward``:

  - student_batch : [system, problem_image_VIT, question, completion]
  - teacher_batch : [system, problem_image_VIT, question, reference_intro,
                     (VT_image_i)+, transition, completion]

The teacher's privileged channel is **strictly visual-only**: it sees only
the intermediate VT images.  Both the text-form ``thought_i`` traces and
the ground-truth final ``answer`` are deliberately omitted, so the
teacher--student information gap isolates the visual generation pathway.

Only the *completion* block has ``loss=1`` on either side.  Once the model
runs a forward pass with ``return_logits=True``, the returned ``logits``
tensor has shape ``[N_completion_ce, V]`` — the same shape for student
and teacher because they share the identical completion — and can be
fed directly into ``generalized_jsd_loss`` without any additional masking.

We intentionally keep batch size = 1 here.  Gradient accumulation and
data-parallel batching happen at the trainer level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .data_utils import (
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)


# System / transition prompts — kept as module constants so tests stay
# deterministic.  They are pure strings fed through the standard tokenizer.

VLM_THINK_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. For text-based "
    "thinking, enclose the process within <think> </think>, e.g. "
    "<think> thinking process here </think>. For visual thinking, "
    "enclose the content within <image_start> </image_end>, e.g. "
    "<image_start> thinking image here </image_end>. Finally conclude "
    "with the final answer wrapped in <answer></answer> tags, i.e. "
    "<answer> answer here </answer>."
)

TEACHER_REFERENCE_INTRO = (
    "The following images are privileged visual references that depict "
    "the intermediate visual thoughts on the path to the correct answer. "
    "Use them silently as grounding context; do not describe or echo "
    "them."
)
TEACHER_TRANSITION = (
    "Now, using your own independent reasoning, answer the problem "
    "above. Think step by step."
)


# --------------------------------------------------------------------- #
# Lightweight builder
# --------------------------------------------------------------------- #


@dataclass
class PackBuilderConfig:
    """Parameters mirrored from ``DataConfig`` / ``PackedDataset`` init."""

    vit_patch_size: int = 14
    max_num_patch_per_side: int = 70
    vae_image_downsample: int = 16
    max_latent_size: int = 32
    text_cond_dropout_prob: float = 0.0
    vit_cond_dropout_prob: float = 0.0
    vae_cond_dropout_prob: float = 0.0
    interpolate_pos: bool = False
    use_flex: bool = False
    max_num_tokens: int = 16384  # used when use_flex=True to size padding


def _empty_status() -> Dict[str, Any]:
    return dict(
        curr=0,
        sample_lens=[],
        packed_position_ids=[],
        nested_attention_masks=[],
        split_lens=[],
        attn_modes=[],
        packed_text_ids=[],
        packed_text_indexes=[],
        packed_label_ids=[],
        ce_loss_indexes=[],
        ce_loss_weights=[],
        vae_image_tensors=[],
        packed_latent_position_ids=[],
        vae_latent_shapes=[],
        packed_vae_token_indexes=[],
        packed_timesteps=[],
        mse_loss_indexes=[],
        packed_vit_tokens=[],
        vit_token_seqlens=[],
        packed_vit_position_ids=[],
        packed_vit_token_indexes=[],
    )


class OPSDPackBuilder:
    """Minimal single-sample packer for Visual-OPSD.

    Mirrors ``PackedDataset.pack_sequence`` but:

      * Does not maintain a data source / buffer / iterator.
      * Only handles one sample per ``build_*`` call.
      * Emits a dict ready for ``Bagel.forward`` (after a ``.cuda(device)``).
    """

    def __init__(
        self,
        config: PackBuilderConfig,
        special_tokens: Dict[str, int],
        system_prompt: Optional[str] = None,
        reference_intro: str = TEACHER_REFERENCE_INTRO,
        transition_prompt: str = TEACHER_TRANSITION,
    ) -> None:
        self.config = config
        self.bos_token_id = special_tokens["bos_token_id"]
        self.eos_token_id = special_tokens["eos_token_id"]
        self.start_of_image = special_tokens["start_of_image"]
        self.end_of_image = special_tokens["end_of_image"]
        self.system_prompt = system_prompt  # optional; None disables system prefix
        self.reference_intro = reference_intro
        self.transition_prompt = transition_prompt
        self.get_flattened_position_ids = (
            get_flattened_position_ids_interpolate
            if config.interpolate_pos
            else get_flattened_position_ids_extrapolate
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def build_student_sample(
        self,
        raw: Dict[str, Any],
        completion_ids: List[int],
        instruction_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Student: [(system?), problem_image, question(+instr), completion].

        ``completion_ids`` should be the raw sampled text token ids — NO
        leading bos, NO trailing eos — exactly what the inference loop
        strips before returning.
        """
        image_tensor_list = [raw["problem_image_tensor"]]
        text_ids_list: List[List[int]] = []
        plan: List[Dict[str, Any]] = []

        # System prompt (optional)
        if self.system_prompt is not None:
            sys_ids = raw.get("_system_ids")
            if sys_ids is None:
                # Caller is expected to have pre-tokenized the system prompt; if
                # missing, pass empty to avoid silent behaviour changes.
                sys_ids = []
            if sys_ids:
                text_ids_list.append(list(sys_ids))
                plan.append(self._text_item(loss=0))

        # Problem image
        plan.append(self._vit_image_item())

        # Question (+ optional instruction suffix)
        q_ids = list(raw["question_ids"])
        if instruction_ids:
            q_ids = q_ids + list(instruction_ids)
        text_ids_list.append(q_ids)
        plan.append(self._text_item(loss=0))

        # Completion — the only loss-bearing block
        text_ids_list.append(list(completion_ids))
        plan.append(self._text_item(loss=1))

        # Tokens used for CE target on this branch:
        # a text block with loss=1 produces (len(text_ids) + 1) CE slots
        # (the leading BOS plus every sampled token; the last slot predicts EOS).
        ce_count = len(completion_ids) + 1

        return self._finalize_sample(
            image_tensor_list=image_tensor_list,
            text_ids_list=text_ids_list,
            plan=plan,
            ce_count=ce_count,
        )

    def build_teacher_sample(
        self,
        raw: Dict[str, Any],
        completion_ids: List[int],
        instruction_ids: Optional[List[int]] = None,
        ref_intro_ids: Optional[List[int]] = None,
        transition_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Teacher: [(system?), problem_image, question, reference_intro,
                    (VT_image_i)+, transition, completion].

        The privileged channel is **strictly visual-only**: the teacher
        conditions on the intermediate VT images alone.  Both the text-form
        ``thought_i`` traces and the ground-truth final ``answer`` are
        deliberately omitted (even if the upstream dataset still provides
        ``reference_thoughts_ids`` / ``reference_answer_ids``) so that the
        teacher--student information gap isolates the visual generation
        pathway.  Only the final completion block carries loss — the
        teacher CE is not back-propagated; we just need the logits over
        the completion.
        """
        image_tensor_list: List = [raw["problem_image_tensor"]]
        text_ids_list: List[List[int]] = []
        plan: List[Dict[str, Any]] = []

        # System prompt (optional, matches student)
        if self.system_prompt is not None:
            sys_ids = raw.get("_system_ids") or []
            if sys_ids:
                text_ids_list.append(list(sys_ids))
                plan.append(self._text_item(loss=0))

        # Problem image
        plan.append(self._vit_image_item())

        # Question (+ instruction)
        q_ids = list(raw["question_ids"])
        if instruction_ids:
            q_ids = q_ids + list(instruction_ids)
        text_ids_list.append(q_ids)
        plan.append(self._text_item(loss=0))

        # Reference intro
        if ref_intro_ids:
            text_ids_list.append(list(ref_intro_ids))
            plan.append(self._text_item(loss=0))

        # Strictly visual-only privileged context: append every VT image
        # in order, skipping ``None`` entries.  ``reference_thoughts_ids``
        # and ``reference_answer_ids`` from the upstream dataset are
        # intentionally NOT consumed here so that the teacher's privileged
        # channel stays purely visual.
        vt_tensors = raw.get("reference_vt_tensors", [])
        for vt in vt_tensors:
            if vt is None:
                continue
            image_tensor_list.append(vt)
            plan.append(self._vit_image_item())

        # Transition prompt
        if transition_ids:
            text_ids_list.append(list(transition_ids))
            plan.append(self._text_item(loss=0))

        # Completion — the only loss-bearing block (SAME ids as student)
        text_ids_list.append(list(completion_ids))
        plan.append(self._text_item(loss=1))

        ce_count = len(completion_ids) + 1

        return self._finalize_sample(
            image_tensor_list=image_tensor_list,
            text_ids_list=text_ids_list,
            plan=plan,
            ce_count=ce_count,
        )

    # ------------------------------------------------------------------ #
    # packing primitives (adapted from PackedDataset.pack_sequence)
    # ------------------------------------------------------------------ #

    def _text_item(self, loss: int) -> Dict[str, Any]:
        return {
            "type": "text",
            "enable_cfg": 0,
            "loss": loss,
            "special_token_loss": 0,
            "special_token_label": None,
        }

    def _vit_image_item(self) -> Dict[str, Any]:
        return {
            "type": "vit_image",
            "enable_cfg": 0,
            "loss": 0,
            "special_token_loss": 0,
            "special_token_label": None,
        }

    def _pack_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Pack a single sample onto an empty sequence_status and return it."""
        image_tensor_list = list(sample["image_tensor_list"])
        text_ids_list = list(sample["text_ids_list"])
        sequence_plan = list(sample["sequence_plan"])
        status = _empty_status()

        split_lens: List[int] = []
        attn_modes: List[str] = []
        curr = status["curr"]
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                # NOTE: unlike PackedDataset we ignore text_cond_dropout_prob;
                # OPSD does not expose condition-dropout on the student side.

                shifted_text_ids = [self.bos_token_id] + list(text_ids)
                status["packed_text_ids"].extend(shifted_text_ids)
                status["packed_text_indexes"].extend(
                    range(curr, curr + len(shifted_text_ids))
                )
                if item["loss"] == 1:
                    status["ce_loss_indexes"].extend(
                        range(curr, curr + len(shifted_text_ids))
                    )
                    status["ce_loss_weights"].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    status["packed_label_ids"].extend(
                        list(text_ids) + [self.eos_token_id]
                    )
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # <|im_end|>
                status["packed_text_ids"].append(self.eos_token_id)
                status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    status["ce_loss_indexes"].append(curr)
                    status["ce_loss_weights"].append(1.0)
                    status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("causal")
                status["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)
                # enable_cfg=0 here; skip the dropout branch entirely.

                status["packed_text_ids"].append(self.start_of_image)
                status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                vit_tokens = patchify(image_tensor, self.config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                status["packed_vit_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                status["packed_vit_tokens"].append(vit_tokens)
                status["vit_token_seqlens"].append(num_img_tokens)
                status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.config.vit_patch_size,
                        max_num_patches_per_side=self.config.max_num_patch_per_side,
                    )
                )

                status["packed_text_ids"].append(self.end_of_image)
                status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    status["ce_loss_indexes"].append(curr)
                    status["ce_loss_weights"].append(1.0)
                    status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("full")
                status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            else:
                raise ValueError(f"Unsupported item type in OPSD builder: {item['type']}")

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        status["curr"] = curr
        status["sample_lens"].append(sample_lens)
        if not self.config.use_flex:
            status["nested_attention_masks"].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            status["split_lens"].extend(split_lens)
            status["attn_modes"].extend(attn_modes)

        return status

    def _finalize_sample(
        self,
        image_tensor_list: List[torch.Tensor],
        text_ids_list: List[List[int]],
        plan: List[Dict[str, Any]],
        ce_count: int,
    ) -> Dict[str, Any]:
        sample = {
            "image_tensor_list": image_tensor_list,
            "text_ids_list": text_ids_list,
            "sequence_plan": plan,
        }
        status = self._pack_one(sample)
        batch = self._status_to_batch(status)
        batch["completion_ce_count"] = ce_count
        return batch

    def _status_to_batch(self, status: Dict[str, Any]) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(
            sequence_length=sum(status["sample_lens"]),
            sample_lens=status["sample_lens"],
            packed_text_ids=torch.tensor(status["packed_text_ids"], dtype=torch.long),
            packed_text_indexes=torch.tensor(
                status["packed_text_indexes"], dtype=torch.long
            ),
            packed_position_ids=torch.tensor(
                status["packed_position_ids"], dtype=torch.long
            ),
        )
        if not self.config.use_flex:
            data["nested_attention_masks"] = status["nested_attention_masks"]
        else:
            sequence_len = data["sequence_length"]
            pad_len = max(self.config.max_num_tokens - sequence_len, 0)
            data["split_lens"] = status["split_lens"] + ([pad_len] if pad_len else [])
            data["attn_modes"] = status["attn_modes"] + (["causal"] if pad_len else [])
            data["sample_lens"] = data["sample_lens"] + ([pad_len] if pad_len else [])

        if status["packed_vit_tokens"]:
            data["packed_vit_tokens"] = torch.cat(status["packed_vit_tokens"], dim=0)
            data["packed_vit_position_ids"] = torch.cat(
                status["packed_vit_position_ids"], dim=0
            )
            data["packed_vit_token_indexes"] = torch.tensor(
                status["packed_vit_token_indexes"], dtype=torch.long
            )
            data["vit_token_seqlens"] = torch.tensor(
                status["vit_token_seqlens"], dtype=torch.int
            )

        if status["packed_label_ids"]:
            data["packed_label_ids"] = torch.tensor(
                status["packed_label_ids"], dtype=torch.long
            )
            data["ce_loss_indexes"] = torch.tensor(
                status["ce_loss_indexes"], dtype=torch.long
            )
            data["ce_loss_weights"] = torch.tensor(
                status["ce_loss_weights"], dtype=torch.float32
            )
        return data


# --------------------------------------------------------------------- #
# Helper to move a packed batch to GPU cleanly
# --------------------------------------------------------------------- #


_TO_DEVICE_KEYS = (
    "packed_text_ids",
    "packed_text_indexes",
    "packed_position_ids",
    "packed_vit_tokens",
    "packed_vit_position_ids",
    "packed_vit_token_indexes",
    "vit_token_seqlens",
    "packed_label_ids",
    "ce_loss_indexes",
    "ce_loss_weights",
)


def packed_batch_to_device(
    batch: Dict[str, Any], device, non_blocking: bool = True
) -> Dict[str, Any]:
    """Move all tensor fields to ``device`` in-place (and return the dict).

    ``nested_attention_masks`` is handled as a list of tensors.
    """
    for k in _TO_DEVICE_KEYS:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=non_blocking)
    if "nested_attention_masks" in batch:
        batch["nested_attention_masks"] = [
            m.to(device, non_blocking=non_blocking)
            for m in batch["nested_attention_masks"]
        ]
    return batch


def get_completion_ce_count(batch: Dict[str, Any]) -> int:
    """Number of CE rows in the packed batch (== completion length + 1)."""
    return int(batch.get("completion_ce_count", 0))
