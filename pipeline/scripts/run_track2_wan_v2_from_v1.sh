#!/usr/bin/env bash
# Run the documented Track 2 Wan trajectory continuation from the v1 best checkpoint.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

export WAM_WAN_OUTPUT="${WAM_WAN_OUTPUT:-artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2-from-v1}"
export WAM_WAN_INIT_CHECKPOINT="${WAM_WAN_INIT_CHECKPOINT:-artifacts/checkpoints/track2-wan-adjust-bottle-v1/best/track2_wan_lora.pt}"
export WAM_WAN_TRAJECTORY_CONDITIONED=1
export WAM_WAN_TRAJECTORY_HIDDEN_DIM="${WAM_WAN_TRAJECTORY_HIDDEN_DIM:-256}"
export WAM_WAN_TRAJECTORY_WARMUP_STEPS="${WAM_WAN_TRAJECTORY_WARMUP_STEPS:-1000}"
export WAM_WAN_HIGH_MOTION_OVERSAMPLE="${WAM_WAN_HIGH_MOTION_OVERSAMPLE:-1.0}"
export WAM_WAN_LEARNING_RATE="${WAM_WAN_LEARNING_RATE:-2e-4}"
export WAM_WAN_INHERITED_LEARNING_RATE="${WAM_WAN_INHERITED_LEARNING_RATE:-2e-5}"
export WAM_WAN_STEPS="${WAM_WAN_STEPS:-30000}"
export WAM_WAN_VALIDATION_INTERVAL="${WAM_WAN_VALIDATION_INTERVAL:-500}"
export WAM_WAN_SAVE_INTERVAL="${WAM_WAN_SAVE_INTERVAL:-500}"
export WAM_WAN_SMOKE_OUTPUT="${WAM_WAN_SMOKE_OUTPUT:-artifacts/smoke/track2-wan-trajectory-v2-from-v1-real}"

[[ -f "$WAM_WAN_INIT_CHECKPOINT" ]] || {
  echo "v1 initialization checkpoint is missing: $WAM_WAN_INIT_CHECKPOINT" >&2
  exit 1
}

exec bash pipeline/scripts/run_track2_wan_formal.sh "$@"
