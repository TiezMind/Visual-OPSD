"""Visual-OPSD Phase 0: KL Diagnostic — GO/NO-GO gate for Visual-OPSD
training.

Measures the distributional gap between teacher (with VT images) and
student (without VT images) using frozen ThinkMorph-7B weights.

For each sample:
  Teacher: [input_image_ViT, question, thought₁, VT_image_ViT, thought₂, answer]
  Student: [input_image_ViT, question, full_text_thought, answer]

Computes per-token KL divergence at text positions.
GO condition: answer_span_kl > 0.05 nats.

Usage:
  torchrun --nproc_per_node=4 scripts/visual_opsd/kl_diagnostic.py \
    --model_path models/ThinkMorph-7B \
    --num_samples 1000 \
    --output_dir results/visual_opsd_kl_diagnostic
"""

import argparse
import io
import json
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
    """Load ThinkMorph-7B model."""
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
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
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
    ckpt_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors") and f != "ae.safetensors"])
    state_dict = {}
    for f in ckpt_files:
        state_dict.update(load_file(os.path.join(model_path, f)))
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).to(torch.bfloat16).eval()
    return model, tokenizer, new_token_ids


def load_diagnostic_samples(data_root, num_samples, seed=42):
    """Load held-out diagnostic samples (deterministic by pid hash).

    Uses a hash-based split to create a reproducible held-out set that
    won't overlap with training data.
    """
    import hashlib
    datasets = ["Visual_Search", "Spatial_Navigation", "Jigsaw_Assembly", "Chart_Refocus"]
    candidates = []
    for ds_name in datasets:
        data_dir = os.path.join(data_root, ds_name, "data")
        if not os.path.exists(data_dir):
            continue
        for f in sorted(os.listdir(data_dir)):
            if not (f.endswith(".parquet") and "train" in f):
                continue
            table = pq.read_table(os.path.join(data_dir, f))
            for i in range(len(table)):
                pid = str(table.column("pid")[i].as_py())
                h = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100
                if h < 5:  # 5% held-out partition
                    candidates.append((ds_name, os.path.join(data_dir, f), i))

    rng = np.random.RandomState(seed)
    rng.shuffle(candidates)
    candidates = candidates[:num_samples]

    samples = []
    for ds_name, fpath, idx in candidates:
        table = pq.read_table(fpath)
        row = {col: table.column(col)[idx].as_py() for col in table.column_names}
        row["_dataset"] = ds_name
        samples.append(row)
    return samples


def build_sequence(
    model, tokenizer, new_token_ids, vit_transform,
    problem_image, question, thoughts, vt_images, answer,
    include_vt=True, full_text_thought=None,
):
    """Build a packed sequence for a single sample.

    Args:
        include_vt: If True, build teacher sequence (with VT images).
                    If False, build student sequence (no VT, but keep input image).
        full_text_thought: Curated text-only thought for student path.
    Returns:
        dict with packed tensors ready for model.forward(), plus answer_ce_len.
    """
    bos_id = new_token_ids["bos_token_id"]
    eos_id = new_token_ids["eos_token_id"]
    start_of_image = new_token_ids["start_of_image"]
    end_of_image = new_token_ids["end_of_image"]
    vit_patch_size = 14
    max_patches_per_side = 70

    curr = 0
    packed_text_ids = []
    packed_text_indexes = []
    packed_position_ids = []
    packed_label_ids = []
    ce_loss_indexes = []
    packed_vit_tokens_list = []
    packed_vit_position_ids_list = []
    packed_vit_token_indexes = []
    vit_token_seqlens = []
    split_lens = []
    attn_modes = []
    curr_rope_id = 0

    def add_text_segment(text_ids, has_loss=False):
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

    def add_vit_image(image_pil):
        nonlocal curr, curr_rope_id
        image_tensor = vit_transform(pil_img2rgb(image_pil))
        vit_tokens = patchify(image_tensor, vit_patch_size)
        num_tokens = vit_tokens.shape[0]

        packed_text_ids.append(start_of_image)
        packed_text_indexes.append(curr)
        curr += 1

        packed_vit_token_indexes.extend(range(curr, curr + num_tokens))
        packed_vit_tokens_list.append(vit_tokens)
        vit_token_seqlens.append(num_tokens)
        packed_vit_position_ids_list.append(
            get_flattened_position_ids_extrapolate(
                image_tensor.size(1), image_tensor.size(2),
                vit_patch_size, max_num_patches_per_side=max_patches_per_side
            )
        )
        curr += num_tokens

        packed_text_ids.append(end_of_image)
        packed_text_indexes.append(curr)
        curr += 1

        seg_len = num_tokens + 2
        split_lens.append(seg_len)
        attn_modes.append("full")
        packed_position_ids.extend([curr_rope_id] * seg_len)
        curr_rope_id += 1

    # 1) Input image (always included)
    add_vit_image(problem_image)

    # 2) Question (no loss)
    q_ids = tokenizer.encode(question)
    add_text_segment(q_ids, has_loss=False)

    if include_vt:
        # Teacher path: interleaved text + VT images
        for i, thought in enumerate(thoughts):
            t_ids = tokenizer.encode(thought)
            add_text_segment(t_ids, has_loss=True)
            if i < len(vt_images) and vt_images[i] is not None:
                add_vit_image(vt_images[i])
    else:
        # Student path: no VT images, input image preserved
        full_text = full_text_thought if full_text_thought else " ".join(thoughts)
        t_ids = tokenizer.encode(full_text)
        add_text_segment(t_ids, has_loss=True)

    # 3) Answer
    a_ids = tokenizer.encode(answer)
    answer_ce_start = len(ce_loss_indexes)
    add_text_segment(a_ids, has_loss=True)
    answer_ce_len = len(ce_loss_indexes) - answer_ce_start

    sequence_length = curr
    device = next(model.parameters()).device

    attn_mask = prepare_attention_mask_per_sample(split_lens, attn_modes, device=device)

    data = {
        "sequence_length": sequence_length,
        "sample_lens": [sequence_length],
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


def compute_kl_for_sample(model, tokenizer, new_token_ids, vit_transform, row):
    """Compute per-token KL between teacher and student for one sample."""
    problem_image = Image.open(io.BytesIO(row["problem_image_0"]["bytes"])).convert("RGB")

    thoughts = []
    vt_images = []
    if "resoning_thought_0" in row and row["resoning_thought_0"]:
        thoughts.append(row["resoning_thought_0"])
    if "reasoning_image_0" in row and row["reasoning_image_0"]:
        vt_images.append(Image.open(io.BytesIO(row["reasoning_image_0"]["bytes"])).convert("RGB"))
    if "resoning_thought_1" in row and row["resoning_thought_1"]:
        thoughts.append(row["resoning_thought_1"])

    answer = row.get("answer", "")
    question = row.get("question", "")

    full_text_thought = row.get("full_text_only_thought", None)
    if not thoughts or not answer:
        return None

    try:
        teacher_data = build_sequence(
            model, tokenizer, new_token_ids, vit_transform,
            problem_image, question, thoughts, vt_images, answer,
            include_vt=True,
        )
        student_data = build_sequence(
            model, tokenizer, new_token_ids, vit_transform,
            problem_image, question, thoughts, vt_images, answer,
            include_vt=False, full_text_thought=full_text_thought,
        )
    except Exception as e:
        print(f"Error building sequence for {row.get('pid', '?')}: {e}")
        return None

    answer_ce_len = min(teacher_data.pop("_answer_ce_len"), student_data.pop("_answer_ce_len"))

    model.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        teacher_out = model(**teacher_data, return_logits=True)
        student_out = model(**student_data, return_logits=True)
        torch.cuda.synchronize()
    model.eval()

    if teacher_out["logits"] is None or student_out["logits"] is None:
        return None

    teacher_logits = teacher_out["logits"].float()
    student_logits = student_out["logits"].float()

    min_len = min(teacher_logits.shape[0], student_logits.shape[0])
    if min_len == 0:
        return None

    # Align from the END (answer tokens match at the tail)
    t_logits = teacher_logits[-min_len:]
    s_logits = student_logits[-min_len:]

    t_probs = F.softmax(t_logits, dim=-1)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    per_token_kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)

    # Answer span KL using exact CE-position metadata
    answer_kl = per_token_kl[-answer_ce_len:].mean().item() if answer_ce_len <= min_len else per_token_kl.mean().item()

    return {
        "pid": row.get("pid", ""),
        "task": row.get("_dataset", ""),
        "avg_kl": per_token_kl.mean().item(),
        "max_kl": per_token_kl.max().item(),
        "answer_span_kl": answer_kl,
        "num_text_tokens": min_len,
        "answer_ce_len": answer_ce_len,
        "teacher_ce_tokens": teacher_logits.shape[0],
        "student_ce_tokens": student_logits.shape[0],
        "per_token_kl": per_token_kl.cpu().numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Visual-OPSD KL Diagnostic")
    parser.add_argument("--model_path", type=str, default="models/ThinkMorph-7B")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/visual_opsd_kl_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
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

    if local_rank == 0:
        print(f"Loading {args.num_samples} held-out diagnostic samples from {args.data_root}...")

    samples = load_diagnostic_samples(args.data_root, args.num_samples, seed=args.seed)

    # Distribute samples across ranks
    per_rank = len(samples) // max(world_size, 1)
    start_idx = local_rank * per_rank
    end_idx = start_idx + per_rank if local_rank < world_size - 1 else len(samples)
    my_samples = samples[start_idx:end_idx]

    results = []
    for i, row in enumerate(my_samples):
        if local_rank == 0 and i % 10 == 0:
            print(f"[Rank {local_rank}] Processing {i}/{len(my_samples)}...")
        try:
            result = compute_kl_for_sample(model, tokenizer, new_token_ids, vit_transform, row)
            if result is not None:
                results.append(result)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[Rank {local_rank}] OOM on sample {row.get('pid', '?')}, skipping")
            continue
        except Exception as e:
            print(f"[Rank {local_rank}] Error on sample {row.get('pid', '?')}: {e}")
            continue

    # Save per-rank results
    rank_file = os.path.join(args.output_dir, f"results_rank{local_rank}.json")
    with open(rank_file, "w") as f:
        json.dump(results, f, indent=2)

    if world_size > 1:
        dist.barrier()

    # Aggregate on rank 0
    if local_rank == 0:
        all_results = []
        for r in range(world_size):
            rf = os.path.join(args.output_dir, f"results_rank{r}.json")
            if os.path.exists(rf):
                with open(rf) as f:
                    all_results.extend(json.load(f))

        if not all_results:
            print("NO VALID RESULTS. Cannot proceed.")
            return

        avg_kls = [r["avg_kl"] for r in all_results]
        answer_kls = [r["answer_span_kl"] for r in all_results]
        max_kls = [r["max_kl"] for r in all_results]

        # Per-task breakdown
        tasks = {}
        for r in all_results:
            t = r["task"]
            if t not in tasks:
                tasks[t] = {"avg_kl": [], "answer_span_kl": []}
            tasks[t]["avg_kl"].append(r["avg_kl"])
            tasks[t]["answer_span_kl"].append(r["answer_span_kl"])

        summary = {
            "num_samples": len(all_results),
            "overall_avg_kl": float(np.mean(avg_kls)),
            "overall_median_kl": float(np.median(avg_kls)),
            "overall_std_kl": float(np.std(avg_kls)),
            "overall_max_kl": float(np.max(max_kls)),
            "answer_span_avg_kl": float(np.mean(answer_kls)),
            "answer_span_median_kl": float(np.median(answer_kls)),
            "per_task": {
                t: {
                    "n": len(v["avg_kl"]),
                    "avg_kl": float(np.mean(v["avg_kl"])),
                    "answer_span_kl": float(np.mean(v["answer_span_kl"])),
                }
                for t, v in tasks.items()
            },
        }

        # GO/NO-GO decision
        go = summary["answer_span_avg_kl"] > 0.05
        summary["go_nogo"] = "GO" if go else "NO-GO"
        summary["threshold"] = 0.05

        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("Visual-OPSD KL DIAGNOSTIC RESULTS")
        print("=" * 60)
        print(f"Samples processed: {summary['num_samples']}")
        print(f"Overall avg KL:    {summary['overall_avg_kl']:.4f} nats")
        print(f"Answer span KL:    {summary['answer_span_avg_kl']:.4f} nats")
        print(f"Threshold:         {summary['threshold']} nats")
        print(
            f"Decision:          "
            f"{'GO — Proceed with Visual-OPSD training' if go else 'NO-GO — VT context does not change predictions'}"
        )
        print()
        print("Per-task breakdown:")
        for t, v in summary["per_task"].items():
            print(f"  {t}: n={v['n']}, avg_kl={v['avg_kl']:.4f}, answer_kl={v['answer_span_kl']:.4f}")
        print("=" * 60)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
