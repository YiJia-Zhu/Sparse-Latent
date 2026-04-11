#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CODI_DO_PRINT="${CODI_DO_PRINT:-0}"

CKPT_DIR="${CKPT_DIR:-$DEFAULT_CODI_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp_local}"
LOCAL_TEST_PATH="${LOCAL_TEST_PATH:-$DEFAULT_GSM8K_TEST}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_BASE_MODEL}"

python test.py \
  --data_name gsm8k-local \
  --local_data_path "$LOCAL_TEST_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --seed 11 \
  --model_max_length 512 \
  --bf16 \
  --lora_r 128 --lora_alpha 32 --lora_init \
  --batch_size 128 \
  --greedy True \
  --num_latent 6 \
  --use_prj True \
  --prj_dim 2048 \
  --prj_no_ln False \
  --prj_dropout 0.0 \
  --inf_latent_iterations 6 \
  --inf_num_iterations 1 \
  --remove_eos True \
  --use_lora True \
  --ckpt_dir "$CKPT_DIR"
