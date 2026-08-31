#!/usr/bin/env bash
# Refine the best rollout model for temporal displacement and moving-region texture fidelity.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/go1/bin/python}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"
windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
parent="${WAM_MOTION_TEXTURE_PARENT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-horizon-largebatch-refine-v1/best}"
output="${WAM_MOTION_TEXTURE_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-motion-texture-refine-v1}"
baseline="${WAM_MOTION_TEXTURE_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_largebatch_refine_preview64.json}"
baseline_predictions="${WAM_MOTION_TEXTURE_BASELINE_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_largebatch_refine_preview64_predictions.npz}"
baseline_dynamics="${WAM_MOTION_TEXTURE_BASELINE_DYNAMICS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_largebatch_refine_preview64_dynamics.json}"
candidate="${WAM_MOTION_TEXTURE_CANDIDATE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_refine_preview64.json}"
predictions="${WAM_MOTION_TEXTURE_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_refine_preview64_predictions.npz}"
dynamics="${WAM_MOTION_TEXTURE_DYNAMICS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_refine_preview64_dynamics.json}"
comparison="${WAM_MOTION_TEXTURE_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_refine_preview64_comparison.json}"
promotion="${WAM_MOTION_TEXTURE_PROMOTION:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_refine_preview64_promotion.json}"
statistics="${WAM_AUTOREG_STATISTICS_CACHE:-artifacts/cache/autoregressive_adjust_bottle_train_statistics.npz}"
steps="${WAM_MOTION_TEXTURE_STEPS:-750}"
batch_size="${WAM_MOTION_TEXTURE_BATCH_SIZE:-10}"
learning_rate="${WAM_MOTION_TEXTURE_LEARNING_RATE:-5e-7}"
temporal_weight="${WAM_TEMPORAL_DELTA_WEIGHT:-0.15}"
texture_weight="${WAM_TEXTURE_LAPLACIAN_WEIGHT:-0.08}"
temporal_pool="${WAM_TEMPORAL_DELTA_POOL:-1}"
seed="${WAM_MOTION_TEXTURE_SEED:-20260811}"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows "$windows" --split-manifest "$split" --output "$output" \
  --init-autoregressive-checkpoint "$parent" \
  --steps "$steps" --batch-size "$batch_size" --learning-rate "$learning_rate" --train-rollout-steps 8 \
  --motion-weight 2.5 --motion-threshold 0.03 --horizon-loss-power 1.0 \
  --temporal-delta-weight "$temporal_weight" --texture-laplacian-weight "$texture_weight" \
  --temporal-delta-pool "$temporal_pool" \
  --high-motion-threshold 0.04 --high-motion-oversample-factor 3.0 --high-motion-selection-weight 0.5 \
  --validation-interval 250 --validation-batches 16 --checkpoint-interval 100 \
  --statistics-cache "$statistics" --seed "$seed" --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$output/best" --split validation --samples 64 \
  --high-motion-threshold 0.04 --failure-gif-count 0 \
  --prediction-cache "$predictions" --output "$candidate"

if [[ ! -f "$baseline_dynamics" ]]; then
  "$python_bin" pipeline/scripts/evaluate_prediction_dynamics.py \
    --windows "$windows" --split-manifest "$split" --split validation \
    --prediction-cache "$baseline_predictions" --output "$baseline_dynamics"
fi
"$python_bin" pipeline/scripts/evaluate_prediction_dynamics.py \
  --windows "$windows" --split-manifest "$split" --split validation \
  --prediction-cache "$predictions" --output "$dynamics"

"$python_bin" pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --minimum-relative-improvement 0.0 --output "$comparison"

"$python_bin" - "$baseline" "$candidate" "$baseline_dynamics" "$dynamics" "$comparison" "$promotion" <<'PY'
import json
import os
import sys
from pathlib import Path

baseline_path, candidate_path, baseline_dynamics_path, dynamics_path, comparison_path, output_path = map(
    Path, sys.argv[1:]
)
baseline = json.loads(baseline_path.read_text())
candidate = json.loads(candidate_path.read_text())
baseline_dynamics = json.loads(baseline_dynamics_path.read_text())["metrics"]
candidate_dynamics = json.loads(dynamics_path.read_text())["metrics"]
comparison = json.loads(comparison_path.read_text())

def gain(old, new):
    return (float(old) - float(new)) / float(old)

overall_gain = gain(baseline["model_mae_mean"], candidate["model_mae_mean"])
high_gain = gain(baseline["high_motion_model_mae_mean"], candidate["high_motion_model_mae_mean"])
baseline_late = sum(map(float, baseline["model_mae_by_prediction_frame"][4:])) / 4.0
candidate_late = sum(map(float, candidate["model_mae_by_prediction_frame"][4:])) / 4.0
late_gain = gain(baseline_late, candidate_late)
dynamic_gains = {
    name: gain(baseline_dynamics[name], candidate_dynamics[name])
    for name in (
        "temporal_delta_mae_mean",
        "moving_region_temporal_delta_mae_mean",
        "moving_region_laplacian_mae_mean",
        "motion_magnitude_mae_mean",
    )
}
checks = {
    "overall_regression_within_0.2_percent": overall_gain >= -0.002,
    "high_motion_regression_within_0.2_percent": high_gain >= -0.002,
    "late_horizon_regression_within_0.2_percent": late_gain >= -0.002,
    "temporal_delta_improves_at_least_0.25_percent": dynamic_gains["temporal_delta_mae_mean"] >= 0.0025,
    "moving_region_temporal_delta_does_not_regress": dynamic_gains["moving_region_temporal_delta_mae_mean"] >= 0.0,
    "moving_region_laplacian_improves_at_least_0.25_percent": dynamic_gains["moving_region_laplacian_mae_mean"] >= 0.0025,
    "motion_magnitude_mae_does_not_regress": dynamic_gains["motion_magnitude_mae_mean"] >= 0.0,
}
result = {
    "format": "track2-motion-texture-promotion-v1",
    "baseline": str(baseline_path.resolve()),
    "candidate": str(candidate_path.resolve()),
    "baseline_dynamics": str(baseline_dynamics_path.resolve()),
    "candidate_dynamics": str(dynamics_path.resolve()),
    "comparison": str(comparison_path.resolve()),
    "overall_relative_improvement": overall_gain,
    "high_motion_relative_improvement": high_gain,
    "late_horizon_relative_improvement": late_gain,
    "dynamics_relative_improvements": dynamic_gains,
    "checks": checks,
    "paired_bootstrap_mean_mae_delta_ci95": comparison.get("paired_bootstrap_mean_mae_delta_ci95"),
    "promote": all(checks.values()),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2))
PY
