#!/usr/bin/env bash
# Generate an identity-checked Wan cache, then sweep a direct-flow RGB blend.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

base="${WAM_WAN_BASE_MODEL:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
wan_checkpoint="${WAM_WAN_ENSEMBLE_CHECKPOINT:-artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2/best}"
flow_checkpoint="${WAM_FLOW_ENSEMBLE_CHECKPOINT:-artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best}"
wan_report="${WAM_WAN_ENSEMBLE_REPORT:-artifacts/evaluations/track2_wan_trajectory_v2_euler45_preview64.json}"
wan_cache="${WAM_WAN_ENSEMBLE_CACHE:-artifacts/evaluations/track2_wan_trajectory_v2_euler45_preview64_predictions.npz}"
output="${WAM_WAN_FLOW_ENSEMBLE_OUTPUT:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_preview64.json}"

export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -s "$wan_cache" ]]; then
  conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend track2-wan --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$wan_checkpoint" --wan-base-model "$base" \
    --wan-inference-steps 45 --wan-inference-solver euler --split validation --samples 64 \
    --failure-gif-count 0 --output "$wan_report" --prediction-cache "$wan_cache"
fi

conda run --no-capture-output -n go1 python pipeline/scripts/sweep_wan_flow_ensemble.py \
  --windows "$windows" --split-manifest "$split" --split validation --samples 64 \
  --wan-checkpoint "$wan_checkpoint" --direct-flow-checkpoint "$flow_checkpoint" \
  --wan-base-model "$base" --wan-inference-steps 45 --wan-inference-solver euler \
  --wan-predictions-cache "$wan_cache" --weights 0 0.25 0.5 0.75 1 --output "$output"
