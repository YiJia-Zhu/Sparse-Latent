#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export CODI_MODEL_IMPL=official

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#GPU_ARRAY[@]}}"
MASTER_PORT="${MASTER_PORT:-29517}"

SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/ckpts_official_sparse}"
LOGGING_DIR="${LOGGING_DIR:-$SAVE_DIR/logs}"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_BASE_MODEL}"
RESTORE_PATH="${RESTORE_PATH:-$DEFAULT_CODI_CKPT}"
if [[ -v SELECTOR_PATH ]]; then
  SELECTOR_PATH="${SELECTOR_PATH}"
else
  SELECTOR_PATH="$DEFAULT_SELECTOR_SUMMARY"
fi
SELECTOR_SET="${SELECTOR_SET:-selected_neg}"

EXPT_NAME="${EXPT_NAME:-gsm8k_llama1b_sparse_codi_official_multigpu}"
DATA_NAME="${DATA_NAME:-icot}"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-$DEFAULT_GSM8K_TRAIN}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-8e-4}"
NUM_LATENT="${NUM_LATENT:-6}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
MAX_TOKEN_NUM="${MAX_TOKEN_NUM:-200}"
DISTILL_LOSS_FACTOR="${DISTILL_LOSS_FACTOR:-20}"
REF_LOSS_FACTOR="${REF_LOSS_FACTOR:-2.0}"
SEED="${SEED:-11}"
EXP_MODE="${EXP_MODE:-False}"
EXP_DATA_NUM="${EXP_DATA_NUM:-7473}"
PRINT_LOSS="${PRINT_LOSS:-True}"
REPORT_TO="${REPORT_TO:-none}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-False}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-False}"

echo "[Official Sparse CODI MultiGPU] gpus=$CUDA_VISIBLE_DEVICES num_gpus=$NUM_GPUS master_port=$MASTER_PORT"
echo "[Official Sparse CODI MultiGPU] model=$MODEL_PATH"
echo "[Official Sparse CODI MultiGPU] restore=$RESTORE_PATH"
echo "[Official Sparse CODI MultiGPU] selector=$SELECTOR_PATH set=$SELECTOR_SET"
echo "[Official Sparse CODI MultiGPU] local_data_path=$LOCAL_DATA_PATH"
echo "[Official Sparse CODI MultiGPU] output_root=$SAVE_DIR expt=$EXPT_NAME"
echo "[Official Sparse CODI MultiGPU] epochs=$NUM_EPOCHS lr=$LEARNING_RATE exp_mode=$EXP_MODE exp_data_num=$EXP_DATA_NUM"
echo "[Official Sparse CODI MultiGPU] per_device_batch=$PER_DEVICE_BATCH_SIZE grad_accum=$GRAD_ACCUM"
echo "[Official Sparse CODI MultiGPU] effective_global_batch=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "[Official Sparse CODI MultiGPU] save_strategy=$SAVE_STRATEGY save_steps=$SAVE_STEPS save_total_limit=$SAVE_TOTAL_LIMIT"
echo "[Official Sparse CODI MultiGPU] resume_checkpoint=${RESUME_CHECKPOINT:-<auto-latest-or-empty>}"

export CUDA_VISIBLE_DEVICES
export RESUME_CHECKPOINT

EXTRA_DATA_ARGS=()
if [[ -n "$LOCAL_DATA_PATH" ]]; then
  EXTRA_DATA_ARGS+=(--local_data_path "$LOCAL_DATA_PATH")
fi

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  train.py \
  --output_dir "$SAVE_DIR" \
  --expt_name "$EXPT_NAME" \
  --logging_dir "$LOGGING_DIR" \
  --logging_steps "$LOGGING_STEPS" \
  --model_name_or_path "$MODEL_PATH" \
  --data_name "$DATA_NAME" \
  --seed "$SEED" \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --bf16 \
  --num_train_epochs "$NUM_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --max_grad_norm 2.0 \
  --use_lora True \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --save_strategy "$SAVE_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_safetensors "$SAVE_SAFETENSORS" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --do_train \
  --report_to "$REPORT_TO" \
  --num_latent "$NUM_LATENT" \
  --logging_strategy "steps" \
  --use_prj True \
  --prj_dim 2048 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --remove_eos True \
  --distill_loss_factor "$DISTILL_LOSS_FACTOR" \
  --print_ref_model_stats False \
  --max_token_num "$MAX_TOKEN_NUM" \
  --ref_loss_factor "$REF_LOSS_FACTOR" \
  --exp_mode "$EXP_MODE" \
  --exp_data_num "$EXP_DATA_NUM" \
  --print_loss "$PRINT_LOSS" \
  --restore_from "$RESTORE_PATH" \
  --selective_align_path "$SELECTOR_PATH" \
  --selective_align_set "$SELECTOR_SET" \
  --ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS" \
  "${EXTRA_DATA_ARGS[@]}"
