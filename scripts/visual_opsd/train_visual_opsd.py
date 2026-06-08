"""Visual-OPSD Training — on-policy self-distillation for unified multimodal models.

Each step (per micro-batch, batch-size=1):

  1. Fetch one raw sample (image + question + privileged VT trace).
  2. Sample a full completion from the CURRENT student weights
     (``OnPolicySampler`` + FSDP.summon_full_params).
  3. Build two packed batches sharing that SAME completion:
        student : [image, question, completion]
        teacher : [image, question, <reference_intro>, (VT_image_i)+,
                   <transition>, completion]
     The teacher's privileged channel is STRICTLY VISUAL-ONLY: only the
     intermediate VT images appear.  Text thoughts and the ground-truth
     answer are deliberately omitted so the teacher--student information
     gap isolates the visual generation pathway.
  4. Forward student (grad) and teacher (no_grad).
  5. Loss = ce_weight * CE(student, completion)
         + jsd_weight * generalizedJSD(student_completion_logits,
                                       teacher_completion_logits)
     The paper uses pure JSD: ce_weight=0, jsd_weight=1.
  6. Backward, optimizer step, EMA update.

The JSD is applied over the ENTIRE sampled completion span — not just the
final answer — because that is the set of positions where the student was
on-policy.  Since both packs end with the identical completion block, the
two logits tensors line up 1-to-1 without any extra masking.

Usage
-----
torchrun --nproc_per_node=8 scripts/visual_opsd/train_visual_opsd.py \
    --model_path models/ThinkMorph-7B \
    --dataset_config_file ./data/configs/visual_opsd.yaml \
    --teacher_mode ema --ema_decay 0.995 \
    --jsd_beta 0.5 --jsd_temperature 1.0 --jsd_top_k 256 \
    --jsd_token_clip 0.05 --jsd_weight 1.0 --ce_weight 1.0 \
    --total_steps 8000 --lr 1e-5
"""

from __future__ import annotations

import functools
import os
import sys
from datetime import timedelta
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from transformers.utils.dummy_pt_objects import FalconForQuestionAnswering
import wandb
import yaml
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.data_utils import add_special_tokens
from data.opsd_pack_builder import (
    OPSDPackBuilder,
    PackBuilderConfig,
    TEACHER_REFERENCE_INTRO,
    TEACHER_TRANSITION,
    VLM_THINK_SYSTEM_PROMPT,
    packed_batch_to_device,
)
from data.opsd_paired_dataset import build_opsd_raw_dataset
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from scripts.visual_opsd.on_policy_sampler import (
    OnPolicySampler,
    SamplingConfig,
    broadcast_raw,
)
from scripts.visual_opsd.opsd_loss import compute_student_kd_loss
from train.fsdp_utils import (
    FSDPCheckpoint,
    FSDPConfig,
    fsdp_ema_setup,
    fsdp_ema_update,
    fsdp_wrapper,
    grad_checkpoint_check_fn,
)
from train.train_utils import create_logger, get_latest_ckpt


# --------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------- #


@dataclass
class ModelArguments:
    model_path: str = "models/ThinkMorph-7B"
    llm_qk_norm: bool = True
    tie_word_embeddings: bool = False
    layer_module: str = "Qwen2MoTDecoderLayer"
    max_latent_size: int = 64
    latent_patch_size: int = 2
    vit_patch_size: int = 14
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"
    interpolate_pos: bool = False
    vit_select_layer: int = -2
    vit_rope: bool = False
    text_cond_dropout_prob: float = 0.0
    vae_cond_dropout_prob: float = 0.0
    vit_cond_dropout_prob: float = 0.0


@dataclass
class DataArguments:
    dataset_config_file: str = "data/configs/visual_opsd.yaml"
    prefetch_factor: int = 2
    num_workers: int = 4
    data_seed: int = 42
    max_reference_rounds: int = 8


@dataclass
class TrainingArguments:
    visual_gen: bool = False
    visual_und: bool = True

    results_dir: str = "results/visual_opsd"
    checkpoint_dir: str = "results/visual_opsd/checkpoints"
    wandb_project: str = "Visual-OPSD"
    wandb_name: str = "visual-opsd-run"
    wandb_runid: str = "0"
    wandb_resume: str = "allow"
    wandb_offline: bool = False

    global_seed: int = 4396
    auto_resume: bool = False
    resume_from: Optional[str] = None
    resume_model_only: bool = True
    finetune_from_ema: bool = True
    finetune_from_hf: bool = True

    log_every: int = 10
    save_every: int = 1000
    total_steps: int = 8000

    warmup_steps: int = 200
    lr_scheduler: str = "cosine"
    lr: float = 1e-5
    min_lr: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-15
    ema: float = 0.9999
    max_grad_norm: float = 1.0

    mse_weight: float = 0.0
    ce_weight: float = 1.0
    ce_loss_reweighting: bool = False
    timestep_shift: float = 1.0

    num_replicate: int = 1
    num_shard: int = 4
    sharding_strategy: str = "HYBRID_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False

    freeze_llm: bool = False
    freeze_vit: bool = False
    freeze_vae: bool = True
    freeze_und: bool = False
    copy_init_moe: bool = True
    use_flex: bool = False

    gradient_accumulation_steps: int = 1

    # --- Visual-OPSD: teacher ---
    teacher_mode: str = field(
        default="ema",
        metadata={"help": "self | ema | fixed"},
    )
    ema_decay: float = field(
        default=0.995,
        metadata={
            "help": "EMA decay for the teacher model when teacher_mode=ema. "
            "OPSD uses 0.9-0.99 (much more lag than optimization EMA)."
        },
    )
    use_ema_weight_tracker: bool = field(
        default=False,
        metadata={
            "help": "When teacher_mode != 'ema', also maintain an EMA-of-"
            "weights tracker for eval-time stability.  Costs ~one full "
            "model shard of persistent GPU memory per rank.  When "
            "teacher_mode == 'ema' the EMA model IS the teacher and is "
            "always kept regardless of this flag."
        },
    )
    teacher_cpu_offload: bool = field(
        default=True,
        metadata={
            "help": "If True and teacher_mode='fixed', wrap fixed_teacher "
            "with FSDP CPU offload.  Sharded params live on CPU; PCIe is "
            "paid during the single no-grad teacher forward per micro-step "
            "in exchange for ~one model shard of persistent GPU memory.  "
            "Required on 80 GB GPUs with 3+ FSDP models and "
            "summon_full_params."
        },
    )

    # --- Visual-OPSD: on-policy sampling ---
    sample_max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "TOTAL max-tokens budget for the sampled completion "
            "across all skip-image rounds (not per round).  Set to a "
            "value small enough that the completion, pack, and summon "
            "buffer fit within the per-rank GPU budget."
        },
    )
    sample_temperature: float = 1.0
    sample_do_sample: bool = True
    sample_max_image_skips: int = field(
        default=1,
        metadata={
            "help": "Max number of <image_end> injections to skip image "
            "generation and keep sampling text, for overfit interleaved "
            "policies.  0 disables the fix and preserves the legacy "
            "single-round behaviour.  See "
            "scripts/visual_opsd/on_policy_sampler.py."
        },
    )
    sample_on_rank0_only: bool = field(
        default=False,
        metadata={
            "help": "If True, only global rank 0 materializes the full "
            "unsharded student during FSDP.summon_full_params for "
            "on-policy sampling; the completion is then broadcast to "
            "every other rank, and to preserve (prompt, completion) "
            "consistency rank 0's raw batch is ALSO broadcast to every "
            "other rank before sampling.  Saves ~one full-model BF16 "
            "copy of GPU memory (~29 GB for a 14.5 B MoT backbone) on "
            "every non-rank-0 rank, at the cost of reducing the per-"
            "optimizer-step effective batch size by world_size "
            "(every rank trains on the same sample).  Enable as a last "
            "resort when CPU-offloading the fixed_teacher and the "
            "in-place BF16 summon patch are not enough to fit the "
            "training step within the per-GPU budget, and compensate "
            "for the smaller effective batch by increasing "
            "``gradient_accumulation_steps``."
        },
    )
    use_system_prompt: bool = True
    use_reference_intro: bool = True
    use_transition_prompt: bool = True
    instruction_suffix: str = (
        ""  # e.g. " Let's think step by step." — appended to question text
    )

    # --- Visual-OPSD: forward memory budget ---
    max_forward_tokens: int = field(
        default=10240,
        metadata={
            "help": "Maximum total packed-sequence length (in tokens) for any "
            "single forward pass.  When the teacher batch (which includes "
            "reference thoughts, VT images, and the sampled completion) "
            "exceeds this limit, the completion is truncated so the "
            "sequence fits.  The dense nested-attention mask scales as "
            "O(L^2) and is the dominant OOM contributor for long "
            "sequences; this cap keeps memory predictable."
        },
    )

    # --- Visual-OPSD: distillation loss ---
    jsd_weight: float = 1.0
    jsd_beta: float = field(
        default=0.5,
        metadata={"help": "0=fwd-KL, 1=rev-KL, 0.5=symmetric JSD"},
    )
    jsd_temperature: float = 1.0
    jsd_top_k: int = field(
        default=256,
        metadata={"help": "restrict JSD to teacher's top-K tokens (0=off)"},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "per-token JSD clip (OPSD default 0.05)"},
    )
    jsd_loss_kind: str = field(
        default="jsd",
        metadata={"help": "'jsd' or 'tinker'"},
    )

    noise_vt: bool = field(
        default=False,
        metadata={
            "help": "Replace teacher's privileged VT (visual thought) image "
            "tensors with random Gaussian noise of the same shape.  The "
            "teacher still sees the reference text thoughts and answer; only "
            "the interleaved images become uninformative.  Used as an "
            "ablation to isolate the contribution of visual privileged "
            "information in Visual-OPSD distillation."
        },
    )


# --------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------- #


def build_bagel_model(model_args, training_args):
    llm_config = Qwen2Config.from_json_file(
        os.path.join(model_args.model_path, "llm_config.json")
    )
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_args.model_path, "vit_config.json")
    )
    vit_config.num_hidden_layers = (
        vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
    )
    vit_config.rope = model_args.vit_rope
    vit_model = SiglipVisionModel(vit_config)

    config = BagelConfig(
        visual_gen=False,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model if training_args.visual_und else None, config)
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    return model, llm_config, vit_config


# --------------------------------------------------------------------- #
# Raw sample iterator helper
# --------------------------------------------------------------------- #


def _raw_identity_collate(batch):
    """IterableDataset loader emits single-sample lists; unwrap to a dict."""
    assert len(batch) == 1
    return batch[0]


def _move_image_tensor(sample: Dict[str, Any], device) -> Dict[str, Any]:
    """Move ``problem_image_tensor`` (and VT tensors) to ``device`` eagerly.

    We keep text ids / lists on CPU — they are tiny and are consumed by
    Python-side builders.
    """
    if "problem_image_tensor" in sample and torch.is_tensor(sample["problem_image_tensor"]):
        sample["problem_image_tensor"] = sample["problem_image_tensor"].to(device)
    vts = sample.get("reference_vt_tensors")
    if vts:
        sample["reference_vt_tensors"] = [
            v.to(device) if torch.is_tensor(v) else v for v in vts
        ]
    return sample


# --------------------------------------------------------------------- #
# Optimizer state CPU offload for sampling memory headroom
# --------------------------------------------------------------------- #


def _offload_optim_to_cpu(optimizer: torch.optim.Optimizer) -> None:
    """Move all CUDA optimizer state tensors (AdamW m, v) to CPU.

    After ``optimizer.step()`` lazily creates the FP32 ``exp_avg`` and
    ``exp_avg_sq`` buffers, they collectively consume ~2x the sharded-
    parameter memory per rank.  For a 14.5 B model on 8 GPUs that is
    roughly 14.5 GiB — enough to make ``FSDP.summon_full_params`` OOM
    when allocating the BF16 unshard buffer.

    Call this before the sampling / forward / backward phase so the GPU
    memory is available for the compute-intensive steps.  Pair with
    ``_restore_optim_to_gpu`` before ``optimizer.step()``.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                state[k] = v.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _restore_optim_to_gpu(
    optimizer: torch.optim.Optimizer,
    device: int | torch.device,
) -> None:
    """Move all CPU optimizer state tensors back to *device*.

    Inverse of ``_offload_optim_to_cpu``.  Must be called before
    ``optimizer.step()`` so the state tensors reside on the same device
    as the parameters.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                state[k] = v.to(device, non_blocking=True)
    torch.cuda.synchronize()


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #


def main() -> None:
    assert torch.cuda.is_available()
    nccl_timeout = timedelta(minutes=30)
    dist.init_process_group("nccl", timeout=nccl_timeout)
    # HYBRID_SHARD creates sub-process-groups via init_device_mesh that
    # inherit the default 10-min NCCL timeout, not the one above.  Set
    # the default globally so every new PG picks up 30 min as well.
    if hasattr(torch.distributed.distributed_c10d, "_DEFAULT_PG_NCCL_TIMEOUT"):
        torch.distributed.distributed_c10d._DEFAULT_PG_NCCL_TIMEOUT = nccl_timeout
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()

    logger.info(
        f"Visual-OPSD on-policy | teacher={training_args.teacher_mode} "
        f"β={training_args.jsd_beta} τ={training_args.jsd_temperature} "
        f"topK={training_args.jsd_top_k} clip={training_args.jsd_token_clip} "
        f"kind={training_args.jsd_loss_kind} "
        f"λ_jsd={training_args.jsd_weight} λ_ce={training_args.ce_weight} "
        f"sample_T={training_args.sample_temperature} "
        f"max_new={training_args.sample_max_new_tokens} "
        f"max_fwd={training_args.max_forward_tokens}"
    )

    # -------- resume paths --------
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = (
                training_args.finetune_from_ema if resume_model_only else False
            )
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = (
            training_args.finetune_from_ema if resume_model_only else False
        )

    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # -------- model --------
    model, _, vit_config = build_bagel_model(model_args, training_args)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for p in model.vit_model.parameters():
            p.requires_grad = False

    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    # We keep an EMA model iff (a) it's the teacher (teacher_mode='ema') or
    # (b) the user explicitly asked for an eval-time EMA weight tracker.
    # In teacher_mode in {'self', 'fixed'} the EMA model would otherwise
    # cost ~one full model shard of persistent GPU memory per rank for no
    # training benefit, which is a significant fraction of the 80 GB H800
    # budget once the student (params + FP32 AdamW state), fixed_teacher,
    # and the summon_full_params unshard buffer are accounted for.
    keep_ema_tracker = (
        training_args.teacher_mode == "ema"
        or training_args.use_ema_weight_tracker
    )
    ema_model = deepcopy(model) if keep_ema_tracker else None

    # Fixed teacher = a second frozen model loaded from the same init ckpt.
    fixed_teacher = None
    if training_args.teacher_mode == "fixed":
        fixed_teacher = deepcopy(model)

    if resume_from is None:
        resume_from = model_args.model_path

    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    if fixed_teacher is not None:
        # Pass ema_model=None: fixed_teacher is a single frozen model loaded
        # from the vanilla HF weights; we must not trigger try_load_ckpt's
        # ema branch (which would deepcopy fixed_teacher, re-read
        # model.safetensors from NAS as an "ema replica", load it into the
        # copy, and then discard the whole thing — pure waste of RAM, I/O,
        # and wall-clock time).
        fixed_teacher, _ = FSDPCheckpoint.try_load_ckpt(
            model_args.model_path, logger, fixed_teacher, None,
            resume_from_ema=False,
        )
        for p in fixed_teacher.parameters():
            p.requires_grad = False

    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    if fixed_teacher is not None:
        # Offload fixed_teacher to CPU.  It is forward-only (no gradient,
        # no optimizer state, no EMA update), so PCIe is only paid during
        # the single no_grad forward per micro-step.  In exchange we free
        # ~one model shard of persistent GPU memory, which the student
        # needs for its summon_full_params unshard buffer during
        # on-policy sampling.
        teacher_offload = (
            fsdp_config.cpu_offload or training_args.teacher_cpu_offload
        )
        teacher_fsdp_config = FSDPConfig(
            sharding_strategy=training_args.sharding_strategy,
            backward_prefetch=training_args.backward_prefetch,
            cpu_offload=teacher_offload,
            num_replicate=training_args.num_replicate,
            num_shard=training_args.num_shard,
        )
        fixed_teacher = fsdp_ema_setup(fixed_teacher, teacher_fsdp_config)

        # Defensive: FSDP's CPUOffload has a known gotcha when every
        # parameter of the wrapped module is frozen (requires_grad=False),
        # which is exactly the fixed_teacher case — the init-time
        # "move flat_param to CPU" step can be silently skipped, and the
        # sharded FP32 master weights stay resident on the GPU (for our
        # ~14.5 B model that is ~7 GB of persistent waste per rank).  The
        # forward-time path in ``FlatParamHandle.pre_unshard`` moves the
        # flat_param to the compute device unconditionally if
        # ``offload_params=True``, so we can safely force a move to CPU
        # here after wrap and still have forward work.
        if teacher_offload:
            try:
                from torch.distributed.fsdp import _traversal_utils as _fsdp_traversal
                handles = _fsdp_traversal._get_fsdp_handles(fixed_teacher)
            except Exception:
                handles = []
            moved = 0
            for handle in handles:
                fp = getattr(handle, "flat_param", None)
                if fp is None or not torch.is_tensor(fp):
                    continue
                if fp.device.type == "cuda":
                    fp.data = fp.data.to("cpu")
                    moved += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if dist.get_rank() == 0:
                dev_sample = (
                    handles[0].flat_param.device if handles else "<no-handles>"
                )
                logger.info(
                    f"fixed_teacher FSDP: cpu_offload={teacher_offload}, "
                    f"num_handles={len(handles)}, moved_to_cpu={moved}, "
                    f"first_flat_param_device={dev_sample}"
                )
    fsdp_model = fsdp_wrapper(model, fsdp_config)

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn,
    )

    if dist.get_rank() == 0:
        trainable = sum(p.numel() for p in fsdp_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in fsdp_model.parameters())
        logger.info(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total") # Parameters: 1820.1M trainable / 1822.3M total

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )

    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config
        )

    # -------- dataset --------
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)

    # Inject the user's max_reference_rounds into every group's kwargs so
    # OPSDPairedIterableDataset picks it up.
    for grouped_name in list(dataset_meta.keys()):
        dataset_meta[grouped_name].setdefault(
            "max_reference_rounds", data_args.max_reference_rounds
        )

    raw_dataset = build_opsd_raw_dataset(
        dataset_meta=dataset_meta,
        tokenizer=tokenizer,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        data_status=data_status,
    )
    raw_dataset.set_epoch(data_args.data_seed)
    raw_loader = DataLoader(
        raw_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=False,  # raw envelope has PIL-derived tensors + python lists
        collate_fn=_raw_identity_collate,
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor if data_args.num_workers > 0 else None,
        persistent_workers=data_args.num_workers > 0,
    )

    # -------- samplers + pack builder --------
    sys_prompt_text = (
        VLM_THINK_SYSTEM_PROMPT if training_args.use_system_prompt else None
    )
    sys_prompt_ids = (
        tokenizer.encode(VLM_THINK_SYSTEM_PROMPT, add_special_tokens=False)
        if training_args.use_system_prompt
        else None
    )
    ref_intro_ids = (
        tokenizer.encode(TEACHER_REFERENCE_INTRO, add_special_tokens=False)
        if training_args.use_reference_intro
        else None
    )
    transition_ids = (
        tokenizer.encode(TEACHER_TRANSITION, add_special_tokens=False)
        if training_args.use_transition_prompt
        else None
    )
    instruction_ids = (
        tokenizer.encode(training_args.instruction_suffix, add_special_tokens=False)
        if training_args.instruction_suffix
        else None
    )

    sampler = OnPolicySampler(
        tokenizer=tokenizer,
        vit_transform=None,
        new_token_ids=new_token_ids,
        cfg=SamplingConfig(
            max_new_tokens=training_args.sample_max_new_tokens,
            temperature=training_args.sample_temperature,
            do_sample=training_args.sample_do_sample,
            system_prompt=sys_prompt_text,
            instruction_suffix=(
                training_args.instruction_suffix
                if training_args.instruction_suffix
                else None
            ),
            max_image_skips=training_args.sample_max_image_skips,
        ),
    )

    pack_builder = OPSDPackBuilder(
        config=PackBuilderConfig(
            vit_patch_size=model_args.vit_patch_size,
            max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            max_latent_size=model_args.max_latent_size,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
        ),
        special_tokens=new_token_ids,
        system_prompt=VLM_THINK_SYSTEM_PROMPT if training_args.use_system_prompt else None,
    )

    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()
    if fixed_teacher is not None:
        fixed_teacher.eval()

    grad_accum = training_args.gradient_accumulation_steps
    logger.info(
        f"Training for {training_args.total_steps} steps "
        f"(grad_accum={grad_accum}), starting at {train_step}..."
    )
    data_iter = iter(raw_loader)

    pbar = None
    if dist.get_rank() == 0:
        pbar = tqdm(
            total=training_args.total_steps,
            initial=train_step,
            desc="Visual-OPSD",
            dynamic_ncols=True,
            smoothing=0.05,
        )

    start_time = time()
    curr_step = train_step

    for curr_step in range(train_step, training_args.total_steps):
        optimizer.zero_grad(set_to_none=True)

        # Free AdamW state tensors (m, v) during the sampling + forward
        # + backward phase.  After step 0 these occupy ~14.5 GiB per
        # rank (FP32, 14.5B / 8 GPUs × 2 buffers) and their absence
        # during compute is harmless — only optimizer.step() needs them.
        _offload_optim_to_cpu(optimizer)
        step_ce = 0.0
        step_jsd = 0.0
        step_loss = 0.0
        step_completion_len = 0
        step_data_indexes = []

        for micro in range(grad_accum):
            # Fetch-and-sample loop with retry.  sampler.generate() calls
            # FSDP.summon_full_params() which is a collective allgather:
            # every rank must enter it the same number of times.  We
            # synchronize the break decision via all_reduce so that if
            # ANY rank got an empty completion, ALL ranks retry together;
            # otherwise the retrying ranks issue an allgather that the
            # exited ranks never join, causing a NCCL hang.
            max_retries = 3
            completion_ids: list[int] = []
            raw: Dict[str, Any] = {}
            for _ in range(max_retries):
                try:
                    raw = next(data_iter)
                except StopIteration:
                    data_iter = iter(raw_loader)
                    raw = next(data_iter)

                raw = _move_image_tensor(raw, device)
                if sys_prompt_ids is not None:
                    raw["_system_ids"] = sys_prompt_ids

                if training_args.sample_on_rank0_only:
                    raw = broadcast_raw(raw, src_rank=0, device=device)

                completion_ids = sampler.generate(
                    fsdp_model=fsdp_model,
                    raw=raw,
                    device=device,
                    rank0_only=training_args.sample_on_rank0_only,
                )
                all_ok = torch.tensor(
                    [1 if completion_ids else 0],
                    dtype=torch.int64, device=device,
                )
                dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)
                if all_ok.item() == 1:
                    break

            step_data_indexes.append(raw.get("data_indexes"))
            if not completion_ids:
                # Hard fallback: use a single <eos>-less sentinel token so we
                # still forward/backward and keep FSDP ranks in lockstep.
                # The sampled position is essentially garbage but the sync
                # proceeds.  In practice this path is almost never hit.
                completion_ids = [int(new_token_ids["eos_token_id"])]
                logger.info(
                    f"rank{dist.get_rank()}: fallback completion after "
                    f"{max_retries} empty samples"
                )

            if dist.get_rank() == 0 and curr_step % 50 == 0 and micro == 0:
                preview = tokenizer.decode(completion_ids)
                logger.info(f"[sample@{curr_step}] len={len(completion_ids)} | {preview!r}")
            
            # -------- 2) Build student / teacher packed batches --------
            # Optionally replace privileged VT images with noise (ablation).
            if training_args.noise_vt:
                vts = raw.get("reference_vt_tensors", [])
                raw["reference_vt_tensors"] = [
                    torch.randn_like(v) if v is not None else None
                    for v in vts
                ]

            # Build the teacher batch first to check sequence length; the
            # teacher's packed sequence is always >= the student's because
            # it includes the privileged reference context.
            teacher_batch = pack_builder.build_teacher_sample(
                raw=raw,
                completion_ids=completion_ids,
                instruction_ids=instruction_ids,
                ref_intro_ids=ref_intro_ids,
                transition_ids=transition_ids,
            )

            # Truncate completion to keep the teacher sequence within the
            # forward memory budget.  The dense nested-attention mask
            # scales as O(L^2); capping L avoids OOM on long completions.
            max_fwd = training_args.max_forward_tokens
            teacher_seq_len = teacher_batch["sequence_length"]
            if teacher_seq_len > max_fwd:
                completion_block_len = len(completion_ids) + 2  # BOS + EOS
                teacher_overhead = teacher_seq_len - completion_block_len
                max_compl = max(max_fwd - teacher_overhead - 2, 1)
                if dist.get_rank() == 0:
                    logger.info(
                        f"[trunc@{curr_step}] teacher_seq={teacher_seq_len} > "
                        f"max_fwd={max_fwd}; truncating completion "
                        f"{len(completion_ids)} → {max_compl}"
                    )
                completion_ids = completion_ids[:max_compl]
                teacher_batch = pack_builder.build_teacher_sample(
                    raw=raw,
                    completion_ids=completion_ids,
                    instruction_ids=instruction_ids,
                    ref_intro_ids=ref_intro_ids,
                    transition_ids=transition_ids,
                )

            step_completion_len += len(completion_ids)

            student_batch = pack_builder.build_student_sample(
                raw=raw,
                completion_ids=completion_ids,
                instruction_ids=instruction_ids,
            )

            packed_batch_to_device(student_batch, device)
            packed_batch_to_device(teacher_batch, device)

            # We don't need per-token ce_loss_weights beyond the flat average.
            student_ce_weights = student_batch.pop("ce_loss_weights", None)
            _ = teacher_batch.pop("ce_loss_weights", None)

            # `completion_ce_count` is a pack-builder metadata key used only by
            # analysis / tests; ``Bagel.forward`` does not accept it.  Strip it
            # on both sides before the forward call.
            student_batch.pop("completion_ce_count", None)
            teacher_batch.pop("completion_ce_count", None)

            # Save metadata needed after the forward before we release the
            # batch dicts.
            student_ce_count = len(student_batch.get("ce_loss_indexes", []))

            is_last_micro = micro == grad_accum - 1
            sync_ctx = nullcontext() if is_last_micro else fsdp_model.no_sync()

            with sync_ctx:
                # ---------- Student forward ----------
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    student_out = fsdp_model(**student_batch, return_logits=True)

                # Extract needed values and release the student output dict
                # and batch tensors (especially the L*L attention mask)
                # *before* the teacher forward so memory can be reused.
                ce = student_out.get("ce")
                student_logits = student_out.get("logits")
                del student_out
                # Release the heavy nested_attention_masks tensor.
                student_batch.pop("nested_attention_masks", None)
                student_batch.clear()
                del student_batch
                torch.cuda.empty_cache()

                # ---------- Teacher forward ----------
                if training_args.teacher_mode == "self":
                    teacher_module = fsdp_model
                elif training_args.teacher_mode == "ema":
                    assert ema_model is not None, (
                        "teacher_mode='ema' requires an EMA model"
                    )
                    teacher_module = ema_model
                elif training_args.teacher_mode == "fixed":
                    assert fixed_teacher is not None
                    teacher_module = fixed_teacher
                else:
                    raise ValueError(
                        f"Unknown teacher_mode: {training_args.teacher_mode}"
                    )

                teacher_was_training = teacher_module.training
                teacher_module.train()
                try:
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", enabled=True, dtype=torch.bfloat16
                    ):
                        teacher_out = teacher_module(
                            **teacher_batch, return_logits=True
                        )
                finally:
                    if not teacher_was_training:
                        teacher_module.eval()

                # Extract teacher logits and free teacher output + batch.
                teacher_logits = teacher_out.get("logits")
                del teacher_out
                teacher_batch.pop("nested_attention_masks", None)
                teacher_batch.clear()
                del teacher_batch
                torch.cuda.empty_cache()

                # ---------- CE on student ----------
                ce_val = 0.0
                if ce is not None:
                    if student_ce_weights is not None and training_args.ce_loss_reweighting:
                        ce_scalar = (ce * student_ce_weights).sum() / max(
                            student_ce_weights.sum().item(), 1.0
                        )
                    else:
                        ce_scalar = ce.sum() / max(student_ce_count, 1)
                    ce_val = ce_scalar.item()
                else:
                    ce_scalar = torch.zeros((), device=device)

                # ---------- JSD on full completion ----------
                # Pass logits in the native dtype (bf16); generalized_jsd_loss
                # applies top_k *before* the float32 upcast so only the
                # top-k slice is ever materialised in FP32.
                jsd_val = 0.0
                jsd_scalar = torch.zeros((), device=device)
                if (
                    student_logits is not None
                    and teacher_logits is not None
                    and student_logits.shape[0] > 0
                    and teacher_logits.shape[0] > 0
                ):
                    n = min(student_logits.shape[0], teacher_logits.shape[0])
                    jsd_scalar = compute_student_kd_loss(
                        student_logits[:n],
                        teacher_logits[:n],
                        sampled_token_ids=None,
                        labels=None,
                        loss_kind=training_args.jsd_loss_kind,
                        beta=training_args.jsd_beta,
                        temperature=training_args.jsd_temperature,
                        top_k=(
                            training_args.jsd_top_k
                            if training_args.jsd_top_k > 0
                            else None
                        ),
                        token_clip=(
                            training_args.jsd_token_clip
                            if training_args.jsd_token_clip > 0
                            else None
                        ),
                    )
                    jsd_val = jsd_scalar.item()

                del teacher_logits, student_logits

                micro_loss = (
                    training_args.ce_weight * ce_scalar
                    + training_args.jsd_weight * jsd_scalar
                )
                (micro_loss / grad_accum).backward()

            step_ce += ce_val
            step_jsd += jsd_val
            step_loss += micro_loss.item()

        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        _restore_optim_to_gpu(optimizer, device)
        optimizer.step()
        scheduler.step()
        if ema_model is not None:
            fsdp_ema_update(
                ema_model,
                fsdp_model,
                decay=(
                    training_args.ema_decay
                    if training_args.teacher_mode == "ema"
                    else training_args.ema
                ),
            )

        avg_ce = step_ce / max(grad_accum, 1)
        avg_jsd = step_jsd / max(grad_accum, 1)
        avg_loss = step_loss / max(grad_accum, 1)
        avg_completion_len = step_completion_len / max(grad_accum, 1)

        if pbar is not None:
            pbar.set_postfix(
                ce=f"{avg_ce:.4f}",
                jsd=f"{avg_jsd:.4f}",
                loss=f"{avg_loss:.4f}",
                comp_len=f"{avg_completion_len:.0f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )
            pbar.update(1)

        if curr_step % training_args.log_every == 0 and curr_step > train_step:
            torch.cuda.synchronize()
            elapsed = time() - start_time
            steps_done = training_args.log_every
            sps = steps_done / elapsed if elapsed > 0 else 0.0
            eta_h = (
                (training_args.total_steps - curr_step) / sps / 3600
                if sps > 0
                else float("inf")
            )
            logger.info(
                f"(step={curr_step:07d}) CE: {avg_ce:.4f}, "
                f"JSD: {avg_jsd:.4f}, Loss: {avg_loss:.4f}, "
                f"comp_len: {avg_completion_len:.0f}, "
                f"Steps/Sec: {sps:.2f}, ETA: {eta_h:.1f}h"
            )
            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "ce": avg_ce,
                        "jsd": avg_jsd,
                        "total_loss": avg_loss,
                        "completion_len": avg_completion_len,
                        "lr": optimizer.param_groups[0]["lr"],
                        "total_norm": total_norm.item(),
                        "steps_per_sec": sps,
                    },
                    step=curr_step,
                )
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in step_data_indexes:
            if item is None:
                continue
            dn = item.get("dataset_name")
            if dn is None:
                continue
            if dn not in data_status:
                data_status[dn] = {}
            data_status[dn][item["worker_id"]] = item["data_indexes"]

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            gather_list = [None] * dist.get_world_size()
            dist.all_gather_object(gather_list, data_status)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
            )

    if pbar is not None:
        pbar.close()

    gather_list = [None] * dist.get_world_size()
    dist.all_gather_object(gather_list, data_status)
    FSDPCheckpoint.fsdp_save_ckpt(
        ckpt_dir=training_args.checkpoint_dir,
        train_steps=curr_step,
        model=fsdp_model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        fsdp_config=fsdp_config,
        data_status=gather_list,
    )
    logger.info("Visual-OPSD training finished.")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
