#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/env.sh"

SUITE_TAG="${SUITE_TAG:-fullrun_$(date +%Y%m%d)}"

run_one() {
  local expt_name="$1"
  local selector_path="$2"
  local selector_set="$3"
  local gpu_id="$4"

  CUDA_VISIBLE_DEVICES="$gpu_id" \
  EXPT_NAME="$expt_name" \
  SELECTOR_PATH="$selector_path" \
  SELECTOR_SET="$selector_set" \
  bash "$ROOT_DIR/run_sparse_codi_official.sh"
}

run_one "full_state_${SUITE_TAG}" "" "selected_neg" "${GPU_FULL_STATE:-2}" &
PID1=$!
run_one "sparse_no_neg_${SUITE_TAG}" "$DEFAULT_SELECTOR_SUMMARY" "selected_no_neg" "${GPU_NO_NEG:-4}" &
PID2=$!
run_one "sparse_neg_${SUITE_TAG}" "$DEFAULT_SELECTOR_SUMMARY" "selected_neg" "${GPU_NEG:-7}" &
PID3=$!

wait "$PID1"
wait "$PID2"
wait "$PID3"
