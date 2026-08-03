#!/usr/bin/env bash
# Start the public-data Track 2 RLinf recipe after the local service bridge is ready.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root_dir"
rlinf_root=${RLINF_ROOT:-third_party/WorldArena-2.0/RL_env_benchmark}
resources=${OFFICIAL_RESOURCES:-artifacts/official_resources}
reset_dataset=${RLINF_RESET_DATASET:-artifacts/rlinf_public_reset_adjust_bottle}
bridge_url=${RLINF_BRIDGE_URL:-http://127.0.0.1:18080}

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
