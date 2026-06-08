#!/bin/bash
# Visual-OPSD Sanity Check.
# Quick overfit on small data to verify pipeline works.
# Expected: ~5 minutes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
source .venv/bin/activate

MODEL_PATH=${1:-models/ThinkMorph-7B}

echo "=== R001: SFT Sanity Check ==="
torchrun \
  --nproc_per_node=4 \
  --master_port=29502 \
  scripts/visual_opsd/train_visual_opsd_offline.py \
  --mode sft \
  --model_path ${MODEL_PATH} \
  --dataset_config_file ./data/configs/text_reasoning.yaml \
  --finetune_from_hf True \
  --resume_model_only True \
  --finetune_from_ema True \
  --resume_from ${MODEL_PATH} \
  --results_dir results/sanity_sft \
  --checkpoint_dir results/sanity_sft/checkpoints \
  --lr 1e-4 \
  --num_workers 2 \
  --max_num_tokens 4096 \
  --visual_gen False \
  --total_steps 50 \
  --log_every 5 \
  --save_every 100 \
  --num_shard 4 \
  --expected_num_tokens 4096 \
  --warmup_steps 5 \
  --wandb_offline True \
  --wandb_project Visual-OPSD-sanity \
  --wandb_name sanity-sft \
  --kd_weight 0.0

echo ""
echo "=== Sanity check complete. Check loss convergence in results/sanity_sft/ ==="
