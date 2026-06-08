#!/bin/bash
# Visual-OPSD Baseline: Text-only SFT.
# SFT on ThinkMorph text traces, no KL distillation.
# Expected: ~6 hours on 4× H800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
source .venv/bin/activate

MODEL_PATH=${1:-models/ThinkMorph-7B}
TOTAL_STEPS=${2:-8000}
OUTPUT_NAME=${3:-sft-baseline}

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/visual_opsd/train_visual_opsd_offline.py \
  --mode sft \
  --model_path ${MODEL_PATH} \
  --dataset_config_file ./data/configs/text_reasoning.yaml \
  --finetune_from_hf True \
  --resume_model_only True \
  --finetune_from_ema True \
  --resume_from ${MODEL_PATH} \
  --results_dir results/${OUTPUT_NAME} \
  --checkpoint_dir results/${OUTPUT_NAME}/checkpoints \
  --lr 1e-5 \
  --num_workers 4 \
  --max_num_tokens 10240 \
  --visual_gen False \
  --total_steps ${TOTAL_STEPS} \
  --save_every 1000 \
  --num_shard 4 \
  --wandb_project Visual-OPSD \
  --wandb_name ${OUTPUT_NAME} \
  --wandb_runid 1 \
  --kd_weight 0.0
