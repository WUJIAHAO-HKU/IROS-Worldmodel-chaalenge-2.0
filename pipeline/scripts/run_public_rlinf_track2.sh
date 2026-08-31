#!/usr/bin/env bash
# Start the public-data Track 2 RLinf recipe after the local service bridge is ready.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root_dir"
rlinf_root=${RLINF_ROOT:-third_party/WorldArena-2.0/RL_env_benchmark}
resources=${OFFICIAL_RESOURCES:-artifacts/official_resources}
reset_dataset=${RLINF_RESET_DATASET:-artifacts/rlinf_public_reset_adjust_bottle}
bridge_url=${RLINF_BRIDGE_URL:-http://127.0.0.1:18080}
strict_evaluation=${WAM_STRICT_EVALUATION:-artifacts/evaluations/multisource_flow_unet_formal_v1_validation_all.json}
world_model_checkpoint=${WAM_CHECKPOINT_DIR:-}
world_model_backend=${WAM_BACKEND:-multisource-flow-unet}
# This is intentionally explicit: candidates remain blocked unless their
# full held-out report used this same all-frame RGB threshold.
world_model_accept_mae=${WAM_ACCEPT_MAE:-2.5}

# Preserve caller-provided absolute paths while resolving defaults from the repo root.
resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$root_dir" "$1" ;;
  esac
}

rlinf_root=$(resolve_path "$rlinf_root")
resources=$(resolve_path "$resources")
reset_dataset=$(resolve_path "$reset_dataset")
strict_evaluation=$(resolve_path "$strict_evaluation")

[[ -n "$world_model_checkpoint" ]] || {
  echo "WAM_CHECKPOINT_DIR must identify the exact world-model checkpoint accepted for RLinf" >&2
  exit 2
}
world_model_checkpoint=$(resolve_path "$world_model_checkpoint")

acceptance_args=()
if [[ "$world_model_backend" == "wan-flow-ensemble" ]]; then
  [[ -n "${WAM_WAN_BASE_MODEL:-}" ]] || { echo "WAM_WAN_BASE_MODEL is required for wan-flow-ensemble" >&2; exit 2; }
  acceptance_args+=(--wan-base-model "$(resolve_path "$WAM_WAN_BASE_MODEL")")
fi

python pipeline/scripts/require_strict_world_model_acceptance.py \
  --evaluation "$strict_evaluation" \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir "$world_model_checkpoint" \
  --backend "$world_model_backend" \
  --accept-mae "$world_model_accept_mae" \
  "${acceptance_args[@]}" \
  --output "$world_model_checkpoint/strict_acceptance.json"

python pipeline/scripts/check_official_rlinf_resources.py \
  --resources "$resources" --rlinf-root "$rlinf_root" --reset-dataset "$reset_dataset"

export PYTHONPATH="$root_dir/pipeline:$rlinf_root:$root_dir/third_party/openpi-rlinf-full/src:${PYTHONPATH:-}"
export OPENPI_CKPT_PATH="$resources/pi05_adjust_bottle"
export ROBOTWIN_REWARD_MODEL_PATH="${ROBOTWIN_REWARD_MODEL_PATH:-$resources/reward_model/adjust_bottle/full_weights.pt}"
export T5_MODEL_PATH="${T5_MODEL_PATH:-$resources/reward_model/t5-base}"
export RLINF_RESET_DATASET="$reset_dataset"
export EMBODIED_PATH="$rlinf_root/examples/embodiment"
export REPO_PATH="$rlinf_root"
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ALOHA}

curl --fail --silent --show-error "$bridge_url/health" >/dev/null

python "$rlinf_root/examples/embodiment/train_embodied_agent.py" \
  --config-path "$rlinf_root/examples/embodiment/config" \
  --config-name track2_robotwin_adjust_bottle_http_grpo_openpi_pi05 \
  "env.train.http.server_url=$bridge_url" "$@"
