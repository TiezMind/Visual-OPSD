"""Visual-OPSD offline (cached-teacher) trainer.

This script is used for non-paper-default training paths that do not
need the on-policy student sampler:

  --mode sft       : Text-only SFT baseline (CE loss only).  Reproduces
                     the "Text-only SFT" row of Table 2.
  --mode offline   : Visual-OPSD with a *pre-computed* teacher cache
                     (CE + λ·weighted-KL from cached top-K teacher
                     logprobs).  Useful for fast, lightweight
                     ablations; the paper main result uses the
                     on-policy trainer in ``train_visual_opsd.py``.

Usage:
  torchrun --nproc_per_node=4 scripts/visual_opsd/train_visual_opsd_offline.py \
    --mode offline \
    --model_path models/ThinkMorph-7B \
    --dataset_config_file ./data/configs/visual_opsd_offline.yaml \
    --teacher_cache_dir traces/visual_opsd_teacher_cache \
    --kd_weight 0.5 --kd_temperature 2.0 \
    --total_steps 8000 --lr 1e-5
"""

import functools
import os
import sys
import wandb
import yaml
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)


@dataclass
class ModelArguments:
    model_path: str = field(default="models/ThinkMorph-7B")
    llm_path: str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_qk_norm: bool = True
    tie_word_embeddings: bool = False
    layer_module: str = "Qwen2MoTDecoderLayer"
    vae_path: str = "flux/vae/ae.safetensors"
    vit_path: str = "hf/siglip-so400m-14-980-flash-attn2-navit/"
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
    dataset_config_file: str = "data/configs/visual_opsd_offline.yaml"
    prefetch_factor: int = 2
    num_workers: int = 4
    max_num_tokens_per_sample: int = 16384
    max_num_tokens: int = 10240
    prefer_buffer_before: int = 8192
    max_buffer_size: int = 50
    data_seed: int = 42


@dataclass
class TrainingArguments:
    mode: str = field(default="offline", metadata={"help": "'sft' or 'offline'"})
    visual_gen: bool = False
    visual_und: bool = True

    results_dir: str = "results/visual_opsd_offline"
    checkpoint_dir: str = "results/visual_opsd_offline/checkpoints"
    wandb_project: str = "Visual-OPSD"
    wandb_name: str = "visual-opsd-offline-run"
    wandb_runid: str = "0"
    wandb_resume: str = "allow"
    wandb_offline: bool = False

    global_seed: int = 4396
    auto_resume: bool = False
    resume_from: str = None
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
    expected_num_tokens: int = 10240
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

    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps"})

    # Visual-OPSD (offline) — cached-teacher distillation specifics.
    kd_weight: float = field(default=0.5, metadata={"help": "λ: distillation loss weight"})
    kd_temperature: float = field(default=2.0, metadata={"help": "τ: KL temperature"})
    kd_use_vt_weights: bool = field(default=True, metadata={"help": "Use VT-sensitivity weights"})
    teacher_cache_dir: str = field(default="traces/visual_opsd_teacher_cache",
                                    metadata={"help": "Dir with cached teacher logprobs"})


def load_teacher_cache(cache_dir, logger):
    """Load all cached teacher logprobs into a pid-indexed dict."""
    if not cache_dir or not os.path.exists(cache_dir):
        logger.info(f"No teacher cache found at {cache_dir}")
        return {}

    index_path = os.path.join(cache_dir, "index.pt")
    if not os.path.exists(index_path):
        logger.info(f"No index.pt in {cache_dir}")
        return {}

    index = torch.load(index_path, weights_only=False)
    logger.info(f"Loading teacher cache: {len(index)} samples")

    cache = {}
    loaded_files = set()
    for pid, batch_file in index.items():
        if batch_file not in loaded_files:
            fpath = os.path.join(cache_dir, batch_file)
            batch_data = torch.load(fpath, weights_only=False)
            cache.update(batch_data)
            loaded_files.add(batch_file)

    logger.info(f"Teacher cache loaded: {len(cache)} samples, "
                f"{len(loaded_files)} files")
    return cache


def compute_topk_kl(student_logits, teacher_top_logprobs, teacher_top_indices,
                     vt_weights, temperature, use_vt_weights, device):
    """Compute KL divergence using cached top-K teacher logprobs.

    Args:
        student_logits: [num_tokens, vocab_size] student logits
        teacher_top_logprobs: [num_tokens, K] cached teacher log-probabilities
        teacher_top_indices: [num_tokens, K] cached top-K token indices
        vt_weights: [num_tokens] VT-sensitivity weights
        temperature: KL temperature τ
        use_vt_weights: whether to apply VT-sensitivity weighting
        device: CUDA device
    Returns:
        scalar KL loss
    """
    num_tokens = student_logits.shape[0]
    num_teacher = teacher_top_logprobs.shape[0]
    min_len = min(num_tokens, num_teacher)
    if min_len == 0:
        return torch.tensor(0.0, device=device)

    # Align from end (answer tokens match at tail); cast per-sample to save memory
    s_logits = student_logits[-min_len:].float()
    t_logprobs = teacher_top_logprobs[-min_len:].to(device).float()
    t_indices = teacher_top_indices[-min_len:].to(device).long()
    w = vt_weights[-min_len:].to(device).float()

    # Temperature scaling
    s_logits_scaled = s_logits / temperature

    # Gather student log-probs at teacher's top-K positions
    s_log_probs = F.log_softmax(s_logits_scaled, dim=-1)
    s_topk_log_probs = torch.gather(s_log_probs, dim=-1, index=t_indices)

    # Teacher probs from cached logprobs (also temperature-scaled)
    t_probs = (t_logprobs / temperature).softmax(dim=-1)

    # Per-token KL: sum over top-K vocabulary
    per_token_kl = (t_probs * (t_probs.log() - s_topk_log_probs)).sum(dim=-1)

    if use_vt_weights and w.sum() > 0:
        kl = (w * per_token_kl).sum() / w.sum()
    else:
        kl = per_token_kl.mean()

    return kl * (temperature ** 2)


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
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
    logger.info(f"Mode: {training_args.mode}, λ={training_args.kd_weight}, τ={training_args.kd_temperature}")

    # Load teacher cache (offline distillation mode only)
    teacher_cache = {}
    if training_args.mode == "offline" and training_args.kd_weight > 0:
        teacher_cache = load_teacher_cache(training_args.teacher_cache_dir, logger)
        if not teacher_cache:
            logger.warning("offline mode but no teacher cache — falling back to CE-only")

    # Resume logic
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False

    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Build model (understanding-only, no generation branch)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
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

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # FSDP
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)

    if resume_from is None:
        resume_from = model_args.model_path
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
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
        logger.info(f"Parameters: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

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
            resume_from, optimizer, scheduler, fsdp_config,
        )

    # Dataset
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor if data_args.num_workers > 0 else None,
        persistent_workers=data_args.num_workers > 0,
    )

    fsdp_model.train()
    ema_model.eval()

    grad_accum = training_args.gradient_accumulation_steps
    kd_active = (training_args.mode == "offline" and training_args.kd_weight > 0
                 and len(teacher_cache) > 0)
    logger.info(f"Training for {training_args.total_steps} steps "
                f"(mode={training_args.mode}, kd_active={kd_active}, "
                f"grad_accum={grad_accum}), starting at {train_step}...")

    data_iter = iter(train_loader)

    pbar = None
    if dist.get_rank() == 0:
        pbar = tqdm(
            total=training_args.total_steps,
            initial=train_step,
            desc="Training",
            dynamic_ncols=True,
            smoothing=0.05,
        )

    start_time = time()
    curr_step = train_step

    for curr_step in range(train_step, training_args.total_steps):
        optimizer.zero_grad(set_to_none=True)

        step_ce = 0.0
        step_kd = 0.0
        step_loss = 0.0
        step_data_indexes = []

        for micro in range(grad_accum):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                data = next(data_iter)

            data = data.cuda(device).to_dict()
            data_indexes = data.pop("batch_data_indexes", None)
            _ = data.pop("ce_loss_weights", None)

            if data_indexes:
                items = data_indexes if isinstance(data_indexes, list) else [data_indexes]
                step_data_indexes.extend(items)

            is_last_micro = (micro == grad_accum - 1)
            sync_ctx = nullcontext() if is_last_micro else fsdp_model.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    loss_dict = fsdp_model(**data, return_logits=kd_active)

                micro_loss = torch.tensor(0.0, device=device, requires_grad=True)

                ce = loss_dict["ce"]
                ce_val = 0.0
                if ce is not None:
                    num_local_ce = len(data["ce_loss_indexes"])
                    ce_scalar = ce.sum() / max(num_local_ce, 1)
                    micro_loss = micro_loss + ce_scalar * training_args.ce_weight
                    ce_val = ce_scalar.item()

                kd_val = 0.0
                if kd_active and loss_dict.get("logits") is not None and data_indexes:
                    student_logits = loss_dict["logits"]
                    del loss_dict["logits"]
                    ce_indexes = data["ce_loss_indexes"]
                    sample_lens = data["sample_lens"]

                    cumlen = 0
                    sample_ce_counts = []
                    for slen in sample_lens:
                        mask = (ce_indexes >= cumlen) & (ce_indexes < cumlen + slen)
                        sample_ce_counts.append(mask.sum().item())
                        cumlen += slen

                    offset = 0
                    total_kl = torch.tensor(0.0, device=device)
                    num_kl_samples = 0

                    for i, ce_count in enumerate(sample_ce_counts):
                        if ce_count == 0:
                            continue
                        pid = None
                        if isinstance(data_indexes, list) and i < len(data_indexes):
                            pid = data_indexes[i].get("pid")
                        elif isinstance(data_indexes, dict):
                            pid = data_indexes.get("pid")
                        if pid and pid in teacher_cache:
                            t_info = teacher_cache[pid]
                            sample_kl = compute_topk_kl(
                                student_logits=student_logits[offset:offset + ce_count],
                                teacher_top_logprobs=t_info["teacher_top_logprobs"],
                                teacher_top_indices=t_info["teacher_top_indices"],
                                vt_weights=t_info["vt_weights"],
                                temperature=training_args.kd_temperature,
                                use_vt_weights=training_args.kd_use_vt_weights,
                                device=device,
                            )
                            total_kl = total_kl + sample_kl
                            num_kl_samples += 1
                        offset += ce_count

                    if num_kl_samples > 0:
                        kd_micro = total_kl / num_kl_samples
                        micro_loss = micro_loss + kd_micro * training_args.kd_weight
                        kd_val = kd_micro.item()

                (micro_loss / grad_accum).backward()

            step_ce += ce_val
            step_kd += kd_val
            step_loss += micro_loss.item()

        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        avg_ce = step_ce / grad_accum
        avg_kd = step_kd / grad_accum
        avg_loss = step_loss / grad_accum

        if pbar is not None:
            pbar.set_postfix(
                ce=f"{avg_ce:.4f}", kd=f"{avg_kd:.4f}",
                loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )
            pbar.update(1)

        if curr_step % training_args.log_every == 0 and curr_step > train_step:
            torch.cuda.synchronize()
            elapsed = time() - start_time
            steps_done = training_args.log_every
            steps_per_sec = steps_done / elapsed
            remaining_steps = training_args.total_steps - curr_step
            eta_hours = remaining_steps / steps_per_sec / 3600 if steps_per_sec > 0 else float("inf")

            msg = (f"(step={curr_step:07d}) CE: {avg_ce:.4f}, "
                   f"KD: {avg_kd:.4f}, Loss: {avg_loss:.4f}, "
                   f"Steps/Sec: {steps_per_sec:.2f}, "
                   f"ETA: {eta_hours:.1f}h")
            logger.info(msg)

            if dist.get_rank() == 0:
                wandb.log({
                    "ce": avg_ce,
                    "kd_loss": avg_kd,
                    "total_loss": avg_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "total_norm": total_norm.item(),
                    "steps_per_sec": steps_per_sec,
                }, step=curr_step)
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in step_data_indexes:
            dn = item["dataset_name"]
            if dn not in data_status:
                data_status[dn] = {}
            data_status[dn][item["worker_id"]] = item["data_indexes"]

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)
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

    # Final checkpoint
    if dist.get_rank() == 0:
        gather_list = [None] * dist.get_world_size()
    else:
        gather_list = None
    dist.gather_object(data_status, gather_list, dst=0)
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

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

'''
for lam in 0.1 0.3 0.7 1.0; do
  bash scripts/visual_opsd/run_visual_opsd_offline.sh \
    models/ThinkMorph-7B \
    traces/visual_opsd_teacher_cache $lam 2.0 3000 visual-opsd-offline-lam$lam
done

'''