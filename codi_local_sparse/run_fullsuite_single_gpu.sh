#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    python -c 'import sys
rows=[]
for line in sys.stdin:
    line=line.strip()
    if not line:
        continue
    idx_s, used_s = [x.strip() for x in line.split(",")]
    rows.append((int(idx_s), int(used_s)))
free=[row for row in rows if row[1] < 500]
target=min(free or rows, key=lambda x: x[1])
print(target[0])'
}

GPU_ID="${GPU_ID:-$(pick_gpu)}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-fullsuite_single_gpu_${TIMESTAMP}}"
SUITE_ROOT="${SUITE_ROOT:-$RESULTS_DIR/$SUITE_NAME}"

MODEL_PATH="${MODEL_PATH:-$DEFAULT_BASE_MODEL}"
RESTORE_PATH="${RESTORE_PATH:-$DEFAULT_CODI_CKPT}"
LOCAL_TRAIN_PATH="${LOCAL_TRAIN_PATH:-$DEFAULT_GSM8K_TRAIN}"
LOCAL_TEST_PATH="${LOCAL_TEST_PATH:-$DEFAULT_GSM8K_TEST}"
SELECTOR_SUMMARY="${SELECTOR_SUMMARY:-$DEFAULT_SELECTOR_SUMMARY}"

mkdir -p "$SUITE_ROOT"/{logs,evals,tensorboard,plots}

echo "[suite] gpu_id=$GPU_ID" | tee "$SUITE_ROOT/logs/suite.log"
echo "[suite] suite_root=$SUITE_ROOT" | tee -a "$SUITE_ROOT/logs/suite.log"
echo "[suite] model_path=$MODEL_PATH" | tee -a "$SUITE_ROOT/logs/suite.log"
echo "[suite] restore_path=$RESTORE_PATH" | tee -a "$SUITE_ROOT/logs/suite.log"
echo "[suite] selector_summary=$SELECTOR_SUMMARY" | tee -a "$SUITE_ROOT/logs/suite.log"

python - <<PY > "$SUITE_ROOT/suite_meta.json"
import json
meta = {
    "gpu_id": $GPU_ID,
    "suite_root": "$SUITE_ROOT",
    "model_path": "$MODEL_PATH",
    "restore_path": "$RESTORE_PATH",
    "selector_summary": "$SELECTOR_SUMMARY",
    "local_train_path": "$LOCAL_TRAIN_PATH",
    "local_test_path": "$LOCAL_TEST_PATH",
}
print(json.dumps(meta, indent=2, ensure_ascii=False))
PY

run_plain_llama_eval() {
  local out_dir="$SUITE_ROOT/evals/plain_llama"
  echo "[suite] plain llama eval -> $out_dir" | tee -a "$SUITE_ROOT/logs/suite.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python eval_plain_llama_gsm8k.py \
    --model-path "$MODEL_PATH" \
    --local-test-path "$LOCAL_TEST_PATH" \
    --output-dir "$out_dir" \
    --batch-size 32 \
    --max-samples 0 \
    --max-new-tokens 256 \
    --model-max-length 512 \
    --device cuda 2>&1 | tee "$SUITE_ROOT/logs/plain_llama_eval.log"
}

run_one_method() {
  local tag="$1"
  local selector_path="$2"
  local selector_set="$3"
  local expt_name="${SUITE_NAME}_${tag}"
  local logging_dir="$SUITE_ROOT/tensorboard/$tag"
  local train_log="$SUITE_ROOT/logs/${tag}_train.log"
  local eval_dir="$SUITE_ROOT/evals/$tag"
  local ckpt_dir="$ROOT_DIR/ckpts_official_sparse/$expt_name/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_11"

  echo "[suite] train $tag on gpu $GPU_ID" | tee -a "$SUITE_ROOT/logs/suite.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  EXPT_NAME="$expt_name" \
  MODEL_PATH="$MODEL_PATH" \
  RESTORE_PATH="$RESTORE_PATH" \
  LOCAL_DATA_PATH="$LOCAL_TRAIN_PATH" \
  SELECTOR_PATH="$selector_path" \
  SELECTOR_SET="$selector_set" \
  LOGGING_DIR="$logging_dir" \
  REPORT_TO="tensorboard" \
  PRINT_LOSS="False" \
  NUM_EPOCHS="10" \
  EXP_MODE="False" \
  EXP_DATA_NUM="7473" \
  SAVE_STRATEGY="steps" \
  SAVE_STEPS="200" \
  SAVE_TOTAL_LIMIT="20" \
  LOGGING_STEPS="10" \
  bash "$ROOT_DIR/run_sparse_codi_official.sh" 2>&1 | tee "$train_log"

  echo "[suite] eval $tag from $ckpt_dir" | tee -a "$SUITE_ROOT/logs/suite.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_local_codi_teststyle.py \
    --model-path "$MODEL_PATH" \
    --ckpt-dir "$ckpt_dir" \
    --local-test-path "$LOCAL_TEST_PATH" \
    --output-dir "$eval_dir" \
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
    --seed 11 2>&1 | tee "$SUITE_ROOT/logs/${tag}_eval.log"

  python export_tensorboard_loss_plot.py \
    --event-dir "$logging_dir" \
    --trainer-state "$ckpt_dir/trainer_state.json" \
    --output-path "$SUITE_ROOT/plots/${tag}_loss.png" \
    --title "$tag training loss" 2>&1 | tee "$SUITE_ROOT/logs/${tag}_plot.log"
}

write_summary() {
  python - <<PY
import json
from pathlib import Path

suite_root = Path("$SUITE_ROOT")
rows = []

plain_path = suite_root / "evals" / "plain_llama" / "eval_summary.json"
if plain_path.exists():
    data = json.loads(plain_path.read_text())
    rows.append({
        "method": "plain_llama",
        "accuracy": data.get("accuracy"),
        "mean_generated_tokens": data.get("mean_generated_tokens"),
        "eval_summary": str(plain_path),
    })

for method in ["full_state", "sparse_no_neg", "sparse_neg"]:
    path = suite_root / "evals" / method / "eval_summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    rows.append({
        "method": method,
        "accuracy": data.get("average_accuracy"),
        "mean_generated_tokens": data.get("average_mean_generated_tokens"),
        "eval_summary": str(path),
    })

(suite_root / "suite_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

lines = ["# Full Suite Summary", ""]
for row in rows:
    lines.append(
        f"- {row['method']}: accuracy={row['accuracy']}, mean_generated_tokens={row['mean_generated_tokens']}, eval_summary={row['eval_summary']}"
    )
(suite_root / "suite_summary.md").write_text("\n".join(lines), encoding="utf-8")
PY
}

run_plain_llama_eval
run_one_method "full_state" "" "selected_neg"
run_one_method "sparse_no_neg" "$SELECTOR_SUMMARY" "selected_no_neg"
run_one_method "sparse_neg" "$SELECTOR_SUMMARY" "selected_neg"
write_summary

echo "[suite] done" | tee -a "$SUITE_ROOT/logs/suite.log"
