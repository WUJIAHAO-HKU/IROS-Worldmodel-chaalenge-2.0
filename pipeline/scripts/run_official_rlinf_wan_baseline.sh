#!/usr/bin/env bash
# Verify the published RobotWin Wan checkpoint, then run one GIF, API, and MBRL smoke.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

checkpoint_dir="${WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT:-}"
diffsynth_root="${WAM_OFFICIAL_DIFFSYNTH_ROOT:-artifacts/upstream/diffsynth-studio-runtime-local}"
window="${WAM_OFFICIAL_WAN_WINDOW:-artifacts/adjust_bottle_windows_full/episode5_00000.npz}"
source="${WAN_ROBOTWIN_ADJUST_BOTTLE_SOURCE:-}"
output_dir="${WAM_OFFICIAL_WAN_OUTPUT:-artifacts/evaluations/official_rlinf_wan_adjust_bottle}"
port="${WAM_OFFICIAL_WAN_PORT:-18081}"
token="${WAM_BEARER_TOKEN:-local-dev-token}"
model_version="${WAM_MODEL_VERSION:-official-rlinf-wan-robotwin-adjustbottle}"

[[ -n "$checkpoint_dir" ]] || { echo "WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT is required" >&2; exit 2; }
[[ -n "$source" ]] || { echo "WAN_ROBOTWIN_ADJUST_BOTTLE_SOURCE must be the exact organizer checkpoint URL/revision" >&2; exit 2; }
[[ -f "$window" ]] || { echo "missing Track 2 window: $window" >&2; exit 2; }

mkdir -p "$output_dir"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

conda run --no-capture-output -n go1 python pipeline/scripts/verify_official_rlinf_wan_checkpoint.py \
  --checkpoint-dir "$checkpoint_dir" --diffsynth-root "$diffsynth_root" --source "$source" \
  --output "$output_dir/checkpoint_manifest.json"

common_args=(
  --backend official-rlinf-wan
  --checkpoint-dir "$checkpoint_dir"
  --official-diffsynth-root "$diffsynth_root"
  --official-wan-inference-steps 5
  --device "${WAM_DEVICE:-cuda}"
)

conda run --no-capture-output -n go1 python pipeline/scripts/export_prediction_gif.py \
  --window "$window" "${common_args[@]}" \
  --output "$output_dir/open_loop_comparison.gif"

WAM_BACKEND=official-rlinf-wan \
WAM_CHECKPOINT_DIR="$checkpoint_dir" \
WAM_OFFICIAL_DIFFSYNTH_ROOT="$diffsynth_root" \
WAM_OFFICIAL_WAN_INFERENCE_STEPS=5 \
WAM_BEARER_TOKEN="$token" \
WAM_MODEL_VERSION="$model_version" \
WAM_PORT="$port" \
  conda run --no-capture-output -n go1 python pipeline/scripts/serve.py >"$output_dir/service.log" 2>&1 &
service_pid=$!
cleanup() { kill "$service_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:$port/v1/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "http://127.0.0.1:$port/v1/health" >/dev/null
conda run --no-capture-output -n go1 python pipeline/scripts/contract_test.py \
  --base-url "http://127.0.0.1:$port" --token "$token" --model-version "$model_version"

# This is only a two-round data-contract smoke. It does not authorize RLinf
# training and does not claim an organizer score.
conda run --no-capture-output -n go1 python pipeline/scripts/run_mbrl_smoke.py \
  --window "$window" "${common_args[@]}" --rounds 2 >"$output_dir/mbrl_smoke.json"
