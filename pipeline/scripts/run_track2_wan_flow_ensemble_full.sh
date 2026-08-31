#!/usr/bin/env bash
# Resume-safe full validation of the preselected Wan/direct-flow RGB ensemble.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

base="${WAM_WAN_BASE_MODEL:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
wan_checkpoint="${WAM_WAN_ENSEMBLE_CHECKPOINT:-artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2/best}"
flow_checkpoint="${WAM_FLOW_ENSEMBLE_CHECKPOINT:-artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best}"
ensemble="${WAM_WAN_FLOW_ENSEMBLE_CHECKPOINT:-artifacts/checkpoints/track2-wan-direct-flow-ensemble-v2}"
output="${WAM_WAN_FLOW_ENSEMBLE_EVALUATION:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_all.json}"
resume_cache="${WAM_WAN_FLOW_ENSEMBLE_RESUME_CACHE:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_cache}"
prediction_cache="${WAM_WAN_FLOW_ENSEMBLE_PREDICTION_CACHE:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_predictions.npz}"
progress_output="${WAM_WAN_FLOW_ENSEMBLE_PROGRESS:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_progress.json}"
acceptance_output="${WAM_WAN_FLOW_ENSEMBLE_ACCEPTANCE:-artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_strict_acceptance.json}"
lock_file="${WAM_WAN_FLOW_ENSEMBLE_LOCK:-artifacts/logs/track2_wan45_direct_flow_ensemble_v2.lock}"

export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$lock_file")"
exec 9>"$lock_file"
flock -n 9 || { echo "another Wan/direct-flow full evaluation already owns $lock_file" >&2; exit 75; }

if [[ ! -f "$ensemble/ensemble_config.json" ]]; then
  conda run --no-capture-output -n go1 python pipeline/scripts/package_track2_wan_flow_ensemble.py \
    --wan-checkpoint "$wan_checkpoint" --direct-flow-checkpoint "$flow_checkpoint" --wan-base-model "$base" --output "$ensemble" \
    --wan-weight 0.25 --direct-flow-weight 0.75 --wan-inference-steps 45 --wan-inference-solver euler
fi

conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend wan-flow-ensemble --checkpoint-dir "$ensemble" --wan-base-model "$base" \
  --windows "$windows" --split-manifest "$split" --split validation --samples 682 --seed 0 \
  --accept-mae 1.0 --failure-gif-count 3 --export-worst-gifs \
  --failure-gif-dir "${output%.json}_worst_gifs" --progress-interval 1 \
  --progress-output "$progress_output" --resume-cache-dir "$resume_cache" \
  --prediction-cache "$prediction_cache" --output "$output"

# A completed evaluation is not MBRL authorization. Record a pass only after
# full held-out coverage and the exact packaged model identity are verified.
conda run --no-capture-output -n go1 python pipeline/scripts/require_strict_world_model_acceptance.py \
  --evaluation "$output" --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$ensemble" --backend wan-flow-ensemble --wan-base-model "$base" \
  --accept-mae 1.0 --output "$acceptance_output"
