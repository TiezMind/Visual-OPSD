#!/bin/bash
# Visual-OPSD (offline) — Trace Collection.
# Cache teacher top-256 logprobs + VT-sensitivity weights for all
# training data so the offline trainer can do cheap KD without
# re-running the teacher every step.
# Expected: ~8 hours on 4× H800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
source .venv/bin/activate

MODEL_PATH=${1:-models/ThinkMorph-7B}
OUTPUT_DIR=${2:-traces/visual_opsd_teacher_cache}
NGPU=${3:-4}

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

echo "=== Visual-OPSD Trace Collection ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NGPU}"

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/visual_opsd/collect_traces.py \
  --model_path ${MODEL_PATH} \
  --data_root datasets \
  --output_dir ${OUTPUT_DIR} \
  --top_k 256 \
  --batch_save_size 200
