#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

GPU_ID="${GPU_ID:-7}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-sparse_neg_official_match_${TIMESTAMP}}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_DIR/$RUN_NAME}"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_BASE_MODEL}"
LOCAL_TEST_PATH="${LOCAL_TEST_PATH:-$DEFAULT_GSM8K_TEST}"
LOCAL_TRAIN_PATH="${LOCAL_TRAIN_PATH:-$DEFAULT_GSM8K_AUG_TRAIN}"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$DEFAULT_SELECTOR_SUMMARY}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
SEED="${SEED:-11}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-False}"

mkdir -p "$RUN_ROOT"/{logs,eval,tensorboard,plots}

EXPT_NAME="${RUN_NAME}"
LOGGING_DIR="$RUN_ROOT/tensorboard"
CKPT_DIR="$ROOT_DIR/ckpts_official_sparse/$EXPT_NAME/Llama-3.2-1B-Instruct/ep_${NUM_EPOCHS}/lr_0.0008/seed_${SEED}"

echo "[run] method=sparse_neg gpu=$GPU_ID run_root=$RUN_ROOT" | tee "$RUN_ROOT/logs/run.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
EXPT_NAME="$EXPT_NAME" \
MODEL_PATH="$MODEL_PATH" \
RESTORE_PATH="" \
LOCAL_DATA_PATH="$LOCAL_TRAIN_PATH" \
SELECTOR_PATH="$SELECTOR_SUMMARY" \
SELECTOR_SET="selected_neg" \
LOGGING_DIR="$LOGGING_DIR" \
REPORT_TO="tensorboard" \
PRINT_LOSS="False" \
NUM_EPOCHS="$NUM_EPOCHS" \
SEED="$SEED" \
PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
GRAD_ACCUM="$GRAD_ACCUM" \
LEARNING_RATE="8e-4" \
NUM_LATENT="6" \
MODEL_MAX_LENGTH="512" \
MAX_TOKEN_NUM="200" \
DISTILL_LOSS_FACTOR="20" \
REF_LOSS_FACTOR="1.0" \
EXP_MODE="False" \
EXP_DATA_NUM="200" \
SAVE_STRATEGY="$SAVE_STRATEGY" \
SAVE_STEPS="$SAVE_STEPS" \
SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
SAVE_SAFETENSORS="$SAVE_SAFETENSORS" \
LOGGING_STEPS="10" \
bash "$ROOT_DIR/run_sparse_codi_official.sh" 2>&1 | tee "$RUN_ROOT/logs/train.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_local_codi_teststyle.py \
  --model-path "$MODEL_PATH" \
  --ckpt-dir "$CKPT_DIR" \
  --local-test-path "$LOCAL_TEST_PATH" \
  --output-dir "$RUN_ROOT/eval" \
  --batch-size 128 \
  --max-samples 0 \
  --max-new-tokens 256 \
  --model-max-length 512 \
  --num-latent 6 \
  --inf-latent-iterations 6 \
  --inf-num-iterations 1 \
  --lora-r 128 \
  --lora-alpha 32 \
  --prj-dim 2048 \
  --use-prj \
  --greedy \
  --remove-eos \
  --device cuda \
  --seed "$SEED" 2>&1 | tee "$RUN_ROOT/logs/eval.log"

CUDA_VISIBLE_DEVICES="" python export_tensorboard_loss_plot.py \
  --event-dir "$LOGGING_DIR" \
  --trainer-state "$CKPT_DIR/trainer_state.json" \
  --output-path "$RUN_ROOT/plots/loss.png" \
  --title "sparse_neg training loss" 2>&1 | tee "$RUN_ROOT/logs/plot.log"

echo "[run] done" | tee -a "$RUN_ROOT/logs/run.log"
