#!/bin/bash
# Visual-OPSD-Noise ablation: teacher VT images replaced with Gaussian noise.
#
# Identical to run_visual_opsd.sh except --noise_vt True is passed, which
# replaces each privileged VT image tensor with torch.randn_like(vt)
# before the teacher's packed sequence is built.  Every non-VT element of
# the teacher context (system prompt, problem image, question, reference
# intro, transition prompt) is held fixed; only the privileged VT images
# become uninformative Gaussian noise.  This isolates whether the
# Visual-OPSD gains come from VT semantic content (this run) or from
# generic regularization (the surrounding teacher structure).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 min watchdog heartbeat
source .venv/bin/activate

MODEL_PATH=${1:-models/ThinkMorph-7B}
TEACHER_MODE=${2:-ema}                # self | ema | fixed
JSD_BETA=${3:-0.5}
JSD_TEMP=${4:-1.0}
JSD_WEIGHT=${5:-1.0}
TOTAL_STEPS=${6:-2000}
OUTPUT_NAME=${7:-visual-opsd-noise-${TEACHER_MODE}-beta${JSD_BETA}}

# Extras (optional)
EMA_DECAY=${EMA_DECAY:-0.995}
JSD_TOP_K=${JSD_TOP_K:-256}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
CE_WEIGHT=${CE_WEIGHT:-0.0}
SAMPLE_MAX_NEW=${SAMPLE_MAX_NEW:-1024}
SAMPLE_TEMP=${SAMPLE_TEMP:-1.0}
MAX_FORWARD_TOKENS=${MAX_FORWARD_TOKENS:-10240}
SAMPLE_MAX_IMAGE_SKIPS=${SAMPLE_MAX_IMAGE_SKIPS:-1}

MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=(${ARNOLD_WORKER_0_PORT//,/ })
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
NNODES=${ARNOLD_WORKER_NUM}
NODE_RANK=${ARNOLD_ID}

echo "=== Visual-OPSD-Noise Ablation ==="
echo "Model: ${MODEL_PATH}"
echo "Teacher: ${TEACHER_MODE} (ema_decay=${EMA_DECAY}) — VT replaced with NOISE"
echo "Sampler: T=${SAMPLE_TEMP} max_new=${SAMPLE_MAX_NEW} max_img_skips=${SAMPLE_MAX_IMAGE_SKIPS} max_fwd=${MAX_FORWARD_TOKENS}"
echo "JSD: beta=${JSD_BETA} tau=${JSD_TEMP} topK=${JSD_TOP_K} clip=${JSD_TOKEN_CLIP}"
echo "Weights: ce=${CE_WEIGHT} jsd=${JSD_WEIGHT}"
echo "Steps: ${TOTAL_STEPS}"
echo ""

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/visual_opsd/train_visual_opsd.py \
  --model_path ${MODEL_PATH} \
  --dataset_config_file ./data/configs/visual_opsd.yaml \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema False \
  --resume_from ${MODEL_PATH} \
  --results_dir results/${OUTPUT_NAME} \
  --checkpoint_dir results/${OUTPUT_NAME}/checkpoints \
  --lr 1e-5 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --gradient_accumulation_steps 2 \
  --cpu_offload False \
  --use_flex True \
  --visual_gen False \
  --total_steps ${TOTAL_STEPS} \
  --save_every 500 \
  --num_shard ${NPROC_PER_NODE} \
  --teacher_mode ${TEACHER_MODE} \
  --ema_decay ${EMA_DECAY} \
  --sample_max_new_tokens ${SAMPLE_MAX_NEW} \
  --sample_temperature ${SAMPLE_TEMP} \
  --sample_do_sample True \
  --sample_max_image_skips ${SAMPLE_MAX_IMAGE_SKIPS} \
  --max_forward_tokens ${MAX_FORWARD_TOKENS} \
  --jsd_beta ${JSD_BETA} \
  --jsd_temperature ${JSD_TEMP} \
  --jsd_top_k ${JSD_TOP_K} \
  --jsd_token_clip ${JSD_TOKEN_CLIP} \
  --jsd_weight ${JSD_WEIGHT} \
  --ce_weight ${CE_WEIGHT} \
  --noise_vt True \
  --wandb_project Visual-OPSD \
  --wandb_name ${OUTPUT_NAME} \
  --wandb_runid 5
