#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export CODI_MODEL_IMPL=official
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

EXPT_NAME="${EXPT_NAME:-selected_no_neg_fullrun_ddp}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/ckpts_official_sparse}"
LOGGING_DIR="${LOGGING_DIR:-$SAVE_DIR/logs}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_BASE_MODEL}"
RESTORE_PATH="${RESTORE_PATH:-$DEFAULT_CODI_CKPT}"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-$DEFAULT_GSM8K_TRAIN}"
SELECTOR_PATH="${SELECTOR_PATH:-$DEFAULT_SELECTOR_SUMMARY}"

torchrun \
  --nproc_per_node=2 \
  --master_port 29510 \
  train.py \
  --output_dir "$SAVE_DIR" \
  --expt_name "$EXPT_NAME" \
  --logging_dir "$LOGGING_DIR" \
  --logging_steps 10 \
  --model_name_or_path "$MODEL_PATH" \
  --data_name icot \
  --seed 11 \
  --model_max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --bf16 \
  --num_train_epochs 10 \
  --learning_rate 8e-4 \
  --max_grad_norm 2.0 \
  --use_lora True \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 20 \
  --save_safetensors False \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --do_train \
  --report_to none \
  --num_latent 6 \
  --logging_strategy steps \
  --use_prj True \
  --prj_dim 2048 \
  --prj_dropout 0.0 \
  --distill_loss_div_std True \
  --remove_eos True \
  --distill_loss_factor 20 \
  --print_ref_model_stats False \
  --max_token_num 200 \
  --ref_loss_factor 2.0 \
  --exp_mode False \
  --exp_data_num 7473 \
  --print_loss True \
  --restore_from "$RESTORE_PATH" \
  --local_data_path "$LOCAL_DATA_PATH" \
  --selective_align_path "$SELECTOR_PATH" \
  --selective_align_set selected_no_neg \
  --ddp_find_unused_parameters False
