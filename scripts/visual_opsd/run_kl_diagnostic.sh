#!/bin/bash
# Visual-OPSD Phase 0: KL Diagnostic.
# GO/NO-GO gate — measures if VT context changes ThinkMorph-7B predictions.
# Expected: ~1 hour on 4× H800.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
source .venv/bin/activate

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

MODEL_PATH=${1:-models/ThinkMorph-7B}
NUM_SAMPLES=${2:-1000}
OUTPUT_DIR=${3:-results/visual_opsd_kl_diagnostic}

echo "=== Visual-OPSD KL Diagnostic ==="
echo "Model: ${MODEL_PATH}"
echo "Samples: ${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/visual_opsd/kl_diagnostic.py \
  --model_path ${MODEL_PATH} \
  --data_root datasets \
  --num_samples ${NUM_SAMPLES} \
  --output_dir ${OUTPUT_DIR} \
  --seed 42
