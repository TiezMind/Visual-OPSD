"""Visual-OPSD (offline) — Trace Collection.

Caches teacher top-K logprobs and VT-sensitivity weights so that the
offline trainer (``train_visual_opsd_offline.py``) can use cheap KD
without re-running the teacher every step.

For each training sample, runs ThinkMorph-7B in two modes:
  Teacher: [input_image_ViT, question, thought₁, VT_image_ViT, thought₂, answer]
  No-VT:   [input_image_ViT, question, full_text_thought, answer]

Caches per-sample:
  - Top-256 teacher logprobs (float16)   [num_ce_tokens, 256]
  - Top-256 token indices (int32)         [num_ce_tokens, 256]
  - VT-sensitivity weights (float16)      [num_ce_tokens]
  - Answer CE span length                 int

Usage:
  torchrun --nproc_per_node=4 scripts/visual_opsd/collect_traces.py \
    --model_path models/ThinkMorph-7B \
    --output_dir traces/visual_opsd_teacher_cache \
    --top_k 256
"""

import argparse
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data.data_utils import add_special_tokens, patchify, pil_img2rgb
from data.data_utils import get_flattened_position_ids_extrapolate, prepare_attention_mask_per_sample
from data.transforms import ImageTransform
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


def load_model(model_path, device):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    language_model = Qwen2ForCausalLM(llm_config)

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + (-2)
    vit_config.rope = False
    vit_model = SiglipVisionModel(vit_config)

    config = BagelConfig(
        visual_gen=False, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=None,
        latent_patch_size=2, max_latent_size=64, vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh", interpolate_pos=False, timestep_shift=1.0,
    )
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    from safetensors.torch import load_file
    ckpt_files = sorted([f for f in os.listdir(model_path)
                         if f.endswith(".safetensors") and f != "ae.safetensors"])
    state_dict = {}
    for f in ckpt_files:
        state_dict.update(load_file(os.path.join(model_path, f)))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).to(torch.bfloat16).eval()
    return model, tokenizer, new_token_ids


def build_sequence(
    model, tokenizer, new_token_ids, vit_transform,
    problem_image, question, thoughts, vt_images, answer,
    include_vt=True, full_text_thought=None,
):
    bos_id = new_token_ids["bos_token_id"]
    eos_id = new_token_ids["eos_token_id"]
    start_of_image = new_token_ids["start_of_image"]
    end_of_image = new_token_ids["end_of_image"]

    curr = 0
    packed_text_ids, packed_text_indexes, packed_position_ids = [], [], []
    packed_label_ids, ce_loss_indexes = [], []
    packed_vit_tokens_list, packed_vit_position_ids_list = [], []
    packed_vit_token_indexes, vit_token_seqlens = [], []
    split_lens, attn_modes = [], []
    curr_rope_id = 0

    def add_text(text_ids, has_loss=False):
        nonlocal curr, curr_rope_id
        shifted = [bos_id] + text_ids
        packed_text_ids.extend(shifted)
        packed_text_indexes.extend(range(curr, curr + len(shifted)))
        if has_loss:
            ce_loss_indexes.extend(range(curr, curr + len(shifted)))
            packed_label_ids.extend(text_ids + [eos_id])
        curr += len(shifted)
        packed_text_ids.append(eos_id)
        packed_text_indexes.append(curr)
        curr += 1
        seg_len = len(shifted) + 1
        split_lens.append(seg_len)
        attn_modes.append("causal")
        packed_position_ids.extend(range(curr_rope_id, curr_rope_id + seg_len))
        curr_rope_id += seg_len

    def add_vit(image_pil):
        nonlocal curr, curr_rope_id
        image_tensor = vit_transform(pil_img2rgb(image_pil))
        vit_tokens = patchify(image_tensor, 14)
        n = vit_tokens.shape[0]
        packed_text_ids.append(start_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        packed_vit_token_indexes.extend(range(curr, curr + n))
        packed_vit_tokens_list.append(vit_tokens)
        vit_token_seqlens.append(n)
        packed_vit_position_ids_list.append(
            get_flattened_position_ids_extrapolate(
                image_tensor.size(1), image_tensor.size(2), 14, max_num_patches_per_side=70
            )
        )
        curr += n
        packed_text_ids.append(end_of_image)
        packed_text_indexes.append(curr)
        curr += 1
        seg_len = n + 2
        split_lens.append(seg_len)
        attn_modes.append("full")
        packed_position_ids.extend([curr_rope_id] * seg_len)
        curr_rope_id += 1

    add_vit(problem_image)
    add_text(tokenizer.encode(question), has_loss=False)

    if include_vt:
        for i, thought in enumerate(thoughts):
            add_text(tokenizer.encode(thought), has_loss=True)
            if i < len(vt_images) and vt_images[i] is not None:
                add_vit(vt_images[i])
    else:
        txt = full_text_thought if full_text_thought else " ".join(thoughts)
        add_text(tokenizer.encode(txt), has_loss=True)

    answer_ce_start = len(ce_loss_indexes)
    add_text(tokenizer.encode(answer), has_loss=True)
    answer_ce_len = len(ce_loss_indexes) - answer_ce_start

    device = next(model.parameters()).device

    attn_mask = prepare_attention_mask_per_sample(split_lens, attn_modes, device=device)

    data = {
        "sequence_length": curr,
        "sample_lens": [curr],
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
        "nested_attention_masks": [attn_mask],
        "packed_label_ids": torch.tensor(packed_label_ids, dtype=torch.long, device=device),
        "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long, device=device),
    }
    if packed_vit_tokens_list:
        data["packed_vit_tokens"] = torch.cat(packed_vit_tokens_list, dim=0).to(device)
        data["packed_vit_position_ids"] = torch.cat(packed_vit_position_ids_list, dim=0).to(device)
        data["packed_vit_token_indexes"] = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        data["vit_token_seqlens"] = torch.tensor(vit_token_seqlens, dtype=torch.int, device=device)
    data["_answer_ce_len"] = answer_ce_len
    return data


def process_sample(model, tokenizer, new_token_ids, vit_transform, row, top_k=256):
    """Process one sample: get teacher logprobs and VT-sensitivity weights."""
    problem_image = Image.open(io.BytesIO(row["problem_image_0"]["bytes"])).convert("RGB")
    thoughts = []
    vt_images = []
    if row.get("resoning_thought_0"):
        thoughts.append(row["resoning_thought_0"])
    if row.get("reasoning_image_0"):
        vt_images.append(Image.open(io.BytesIO(row["reasoning_image_0"]["bytes"])).convert("RGB"))
    if row.get("resoning_thought_1"):
        thoughts.append(row["resoning_thought_1"])

    answer = row.get("answer", "")
    question = row.get("question", "")
    full_text = row.get("full_text_only_thought", None)
    if not thoughts or not answer:
        return None

    teacher_data = build_sequence(
        model, tokenizer, new_token_ids, vit_transform,
        problem_image, question, thoughts, vt_images, answer,
        include_vt=True,
    )
    novt_data = build_sequence(
        model, tokenizer, new_token_ids, vit_transform,
        problem_image, question, thoughts, vt_images, answer,
        include_vt=False, full_text_thought=full_text,
    )
    answer_ce_len_t = teacher_data.pop("_answer_ce_len")
    answer_ce_len_s = novt_data.pop("_answer_ce_len")

    model.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        teacher_out = model(**teacher_data, return_logits=True)
        novt_out = model(**novt_data, return_logits=True)
        torch.cuda.synchronize()
    model.eval()
    

    if teacher_out["logits"] is None or novt_out["logits"] is None:
        return None
        
    # teacher_logits和novt_logits的形状通常都是 [T, V]，其中T是序列长度（token数），V是词表大小（vocab size）。
    # 也就是说，每一行对应一个token位置，每一列是词表中对应词的logits。
    teacher_logits = teacher_out["logits"].float()
    novt_logits = novt_out["logits"].float()

    # Align from end (answer tokens match at tail)
    min_len = min(teacher_logits.shape[0], novt_logits.shape[0])
    if min_len == 0:
        return None
    t_logits = teacher_logits[-min_len:]
    n_logits = novt_logits[-min_len:]

    # Top-K teacher logprobs
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    topk_logprobs, topk_indices = torch.topk(t_log_probs, k=top_k, dim=-1)

    # VT-sensitivity weights: per-token KL(teacher_with_VT || no_VT)
    t_probs = F.softmax(t_logits, dim=-1)
    n_log_probs = F.log_softmax(n_logits, dim=-1)
    vt_weights = F.kl_div(n_log_probs, t_probs, reduction="none").sum(dim=-1)

    return {
        "teacher_top_logprobs": topk_logprobs.half().cpu(),      # [min_len, top_k]
        "teacher_top_indices": topk_indices.int().cpu(),           # [min_len, top_k]
        "vt_weights": vt_weights.half().cpu(),                     # [min_len]
        "num_aligned_tokens": min_len,
        "answer_ce_len": min(answer_ce_len_t, answer_ce_len_s),
        "teacher_ce_tokens": teacher_logits.shape[0],
        "student_ce_tokens": novt_logits.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Visual-OPSD Trace Collection")
    parser.add_argument("--model_path", type=str,
                        default="models/ThinkMorph-7B")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="traces/visual_opsd_teacher_cache")
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--batch_save_size", type=int, default=100,
                        help="Save traces in batches of this size")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume — reprocess all samples from scratch")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
        device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        device = 0
        torch.cuda.set_device(device)

    os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if local_rank == 0:
        print(f"Loading model from {args.model_path}...")
    model, tokenizer, new_token_ids = load_model(args.model_path, device)
    vit_transform = ImageTransform(image_stride=14, max_image_size=518, min_image_size=224)

    # Load all training data paths
    datasets = ["Visual_Search", "Spatial_Navigation", "Jigsaw_Assembly", "Chart_Refocus"]
    all_samples = []
    for ds_name in datasets:
        data_dir = os.path.join(args.data_root, ds_name, "data")
        if not os.path.exists(data_dir):
            continue
        for f in sorted(os.listdir(data_dir)):
            if f.endswith(".parquet") and "train" in f:
                all_samples.append((ds_name, os.path.join(data_dir, f)))

    # Flatten: list of (ds_name, file_path, row_idx)
    sample_list = []
    for ds_name, fpath in all_samples:
        table = pq.read_table(fpath)
        for i in range(len(table)):
            sample_list.append((ds_name, fpath, i))

    print(f"Total samples: {len(sample_list)}")

    # --- Resume: scan THIS rank's existing traces to find already-collected keys ---
    # Keys are "{ds_name}/{pid}" (composite) to avoid cross-dataset PID collision.
    # For backward compat, old plain-PID keys are also loaded as-is.
    done_keys = set()
    next_batch_num = 1
    if not args.no_resume:
        my_prefix = f"rank{local_rank}_"
        for fname in sorted(os.listdir(args.output_dir)):
            if not fname.endswith(".pt") or fname == "index.pt":
                continue
            if not fname.startswith(my_prefix):
                continue
            try:
                data = torch.load(os.path.join(args.output_dir, fname),
                                  map_location="cpu", weights_only=False)
                done_keys.update(data.keys())
            except Exception as e:
                print(f"  [Rank {local_rank}] Warning: corrupt cache file {fname}: {e}")
                continue
            if "_batch" in fname:
                try:
                    b = int(fname.replace(".pt", "").split("_batch")[1])
                    next_batch_num = max(next_batch_num, b + 1)
                except ValueError:
                    pass

    print(f"[Rank {local_rank}] Resume: {len(done_keys)} keys cached, "
          f"next batch = {next_batch_num}")

    # Distribute across ranks
    per_rank = len(sample_list) // max(world_size, 1)
    start = local_rank * per_rank
    end = start + per_rank if local_rank < world_size - 1 else len(sample_list)
    my_samples = sample_list[start:end]

    print(f"[Rank {local_rank}] Total samples: {len(sample_list)}, "
          f"this rank: {len(my_samples)}")

    # Process and save in batches
    batch_traces = {}
    saved_count = 0
    skipped = 0
    resumed = 0
    file_cache = {}
    batch_num = next_batch_num

    for idx, (ds_name, fpath, row_idx) in enumerate(my_samples):
        if idx % 50 == 0:
            print(f"[Rank {local_rank}] {idx}/{len(my_samples)} "
                  f"(saved={saved_count}, skipped={skipped}, resumed={resumed})")

        if fpath not in file_cache:
            file_cache[fpath] = pq.read_table(fpath)
        table = file_cache[fpath]
        row = {col: table.column(col)[row_idx].as_py() for col in table.column_names}
        pid = str(row.get("pid", f"{ds_name}_{row_idx}"))
        key = f"{ds_name}/{pid}"

        if key in done_keys or pid in done_keys:
            resumed += 1
            continue

        try:
            result = process_sample(model, tokenizer, new_token_ids, vit_transform, row, args.top_k)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as e:
            if idx < 10:
                print(f"  [Rank {local_rank}] Error on {key}: {e}")
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        batch_traces[key] = result
        saved_count += 1

        if len(batch_traces) >= args.batch_save_size:
            batch_file = os.path.join(
                args.output_dir, f"rank{local_rank}_batch{batch_num}.pt"
            )
            torch.save(batch_traces, batch_file)
            batch_traces = {}
            batch_num += 1

    # Save remaining
    if batch_traces:
        batch_file = os.path.join(args.output_dir, f"rank{local_rank}_resume_final.pt")
        torch.save(batch_traces, batch_file)

    print(f"[Rank {local_rank}] Trace collection complete: "
          f"saved={saved_count}, skipped={skipped}, resumed={resumed}")

    # Build index on rank 0
    if world_size > 1:
        dist.barrier()

    if local_rank == 0:
        index = {}
        for f in sorted(os.listdir(args.output_dir)):
            if not f.endswith(".pt") or f == "index.pt":
                continue
            data = torch.load(os.path.join(args.output_dir, f), weights_only=False)
            for key in data:
                index[key] = f
        torch.save(index, os.path.join(args.output_dir, "index.pt"))
        print(f"Index built: {len(index)} samples across {len(set(index.values()))} files")
        total_bytes = sum(
            os.path.getsize(os.path.join(args.output_dir, f))
            for f in os.listdir(args.output_dir) if f.endswith(".pt")
        )
        print(f"Total cache size: {total_bytes / 1e9:.2f} GB")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
