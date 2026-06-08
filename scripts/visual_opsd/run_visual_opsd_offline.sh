#!/bin/bash
# Visual-OPSD (offline) main training.
# CE + VT-sensitivity-weighted KL from cached teacher logprobs.
# Expected: ~6 hours on 4× H800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

MODEL_PATH=${1:-models/ThinkMorph-7B}
TEACHER_CACHE=${2:-traces/visual_opsd_teacher_cache}
KD_WEIGHT=${3:-0.5}
KD_TEMP=${4:-2.0}
TOTAL_STEPS=${5:-8000}
OUTPUT_NAME=${6:-visual-opsd-offline-lam${KD_WEIGHT}-tau${KD_TEMP}}

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

echo "=== Visual-OPSD (offline) Training ==="
echo "Model: ${MODEL_PATH}"
echo "Teacher cache: ${TEACHER_CACHE}"
echo "λ=${KD_WEIGHT}, τ=${KD_TEMP}, steps=${TOTAL_STEPS}"

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/visual_opsd/train_visual_opsd_offline.py \
  --mode offline \
  --model_path ${MODEL_PATH} \
  --dataset_config_file ./data/configs/visual_opsd_offline.yaml \
  --teacher_cache_dir ${TEACHER_CACHE} \
  --finetune_from_hf True \
  --resume_model_only True \
  --finetune_from_ema False \
  --resume_from ${MODEL_PATH} \
  --results_dir results/${OUTPUT_NAME} \
  --checkpoint_dir results/${OUTPUT_NAME}/checkpoints \
  --lr 1e-5 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --max_num_tokens 10240 \
  --gradient_accumulation_steps 1 \
  --cpu_offload False \
  --visual_gen False \
  --total_steps ${TOTAL_STEPS} \
  --save_every 1000 \
  --num_shard ${NPROC_PER_NODE} \
  --kd_weight ${KD_WEIGHT} \
  --kd_temperature ${KD_TEMP} \
  --kd_use_vt_weights True \
  --wandb_project Visual-OPSD \
  --wandb_name ${OUTPUT_NAME} \
  --wandb_runid 2
