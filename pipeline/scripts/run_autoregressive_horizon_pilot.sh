#!/usr/bin/env bash
# Fine-tune the current autoregressive model with normalized late-horizon loss weighting.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/go1/bin/python}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
parent="${WAM_AUTOREG_HORIZON_PARENT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-motion-rebuild-pilot-v1/best}"
output="${WAM_AUTOREG_HORIZON_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-horizon-pilot-v1}"
baseline="${WAM_AUTOREG_HORIZON_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_pilot_preview64.json}"
candidate="${WAM_AUTOREG_HORIZON_CANDIDATE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_pilot_preview64.json}"
predictions="${WAM_AUTOREG_HORIZON_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_pilot_preview64_predictions.npz}"
comparison="${WAM_AUTOREG_HORIZON_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_pilot_comparison.json}"
promotion="${WAM_AUTOREG_HORIZON_PROMOTION:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_pilot_promotion.json}"
full_evaluation="${WAM_AUTOREG_HORIZON_FULL_EVALUATION:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_validation_all.json}"
full_baseline="${WAM_AUTOREG_HORIZON_FULL_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_validation_all.json}"
full_comparison="${WAM_AUTOREG_HORIZON_FULL_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_vs_parent_full682.json}"
full_promotion="${WAM_AUTOREG_HORIZON_FULL_PROMOTION:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_full_promotion.json}"
full_cache="${WAM_AUTOREG_HORIZON_FULL_CACHE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_validation_cache}"
full_progress="${WAM_AUTOREG_HORIZON_FULL_PROGRESS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_validation_progress.json}"
full_predictions="${WAM_AUTOREG_HORIZON_FULL_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_validation_predictions.npz}"
local_evaluation="${WAM_AUTOREG_HORIZON_LOCAL_EVALUATION:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_local_test_all.json}"
local_cache="${WAM_AUTOREG_HORIZON_LOCAL_CACHE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_local_test_cache}"
local_progress="${WAM_AUTOREG_HORIZON_LOCAL_PROGRESS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_local_test_progress.json}"
local_predictions="${WAM_AUTOREG_HORIZON_LOCAL_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_local_test_predictions.npz}"
statistics="${WAM_AUTOREG_STATISTICS_CACHE:-artifacts/cache/autoregressive_adjust_bottle_train_statistics.npz}"
steps="${WAM_AUTOREG_HORIZON_STEPS:-1500}"
batch_size="${WAM_AUTOREG_HORIZON_BATCH_SIZE:-2}"
learning_rate="${WAM_AUTOREG_HORIZON_LEARNING_RATE:-2e-6}"
horizon_loss_power="${WAM_AUTOREG_HORIZON_LOSS_POWER:-1.0}"
validation_interval="${WAM_AUTOREG_HORIZON_VALIDATION_INTERVAL:-250}"
validation_batches="${WAM_AUTOREG_HORIZON_VALIDATION_BATCHES:-64}"
seed="${WAM_AUTOREG_HORIZON_SEED:-20260809}"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows "$windows" --split-manifest "$split" --output "$output" \
  --init-autoregressive-checkpoint "$parent" \
  --steps "$steps" --batch-size "$batch_size" --learning-rate "$learning_rate" --train-rollout-steps 8 \
  --motion-weight 2.5 --motion-threshold 0.03 --horizon-loss-power "$horizon_loss_power" \
  --high-motion-threshold 0.04 --high-motion-oversample-factor 3.0 \
  --high-motion-selection-weight 0.5 \
  --validation-interval "$validation_interval" --validation-batches "$validation_batches" --checkpoint-interval 100 \
  --statistics-cache "$statistics" --seed "$seed" --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$output/best" --split validation --samples 64 \
  --high-motion-threshold 0.04 --failure-gif-count 0 \
  --prediction-cache "$predictions" --output "$candidate"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --minimum-relative-improvement 0.0 \
  --output "$comparison"

write_promotion() {
  "$python_bin" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import os
import sys
from pathlib import Path

baseline_path, candidate_path, comparison_path, output_path = map(Path, sys.argv[1:5])
stage = sys.argv[5]
baseline = json.loads(baseline_path.read_text())
candidate = json.loads(candidate_path.read_text())
comparison = json.loads(comparison_path.read_text())
baseline_mae = float(baseline["model_mae_mean"])
candidate_mae = float(candidate["model_mae_mean"])
baseline_high = float(baseline["high_motion_model_mae_mean"])
candidate_high = float(candidate["high_motion_model_mae_mean"])
baseline_late = sum(map(float, baseline["model_mae_by_prediction_frame"][4:])) / 4.0
candidate_late = sum(map(float, candidate["model_mae_by_prediction_frame"][4:])) / 4.0
overall_gain = (baseline_mae - candidate_mae) / baseline_mae
high_gain = (baseline_high - candidate_high) / baseline_high
late_gain = (baseline_late - candidate_late) / baseline_late
promote = overall_gain >= -0.002 and high_gain >= -0.002 and late_gain >= 0.01
result = {
    "format": "track2-autoregressive-horizon-pilot-promotion-v1",
    "stage": stage,
    "baseline": str(baseline_path.resolve()),
    "candidate": str(candidate_path.resolve()),
    "comparison": str(comparison_path.resolve()),
    "maximum_overall_relative_regression": 0.002,
    "maximum_high_motion_relative_regression": 0.002,
    "minimum_late_horizon_relative_improvement": 0.01,
    "baseline_late_horizon_mae": baseline_late,
    "candidate_late_horizon_mae": candidate_late,
    "overall_relative_improvement": overall_gain,
    "high_motion_relative_improvement": high_gain,
    "late_horizon_relative_improvement": late_gain,
    "paired_bootstrap_mean_mae_delta_ci95": comparison.get("paired_bootstrap_mean_mae_delta_ci95"),
    "promote": promote,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2))
PY
}

write_promotion "$baseline" "$candidate" "$comparison" "$promotion" preview64

if "$python_bin" - "$promotion" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("promote") else 1)
PY
then
  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$output/best" --split validation --samples 682 \
    --high-motion-threshold 0.04 --accept-mae 1.0 --failure-gif-count 3 \
    --resume-cache-dir "$full_cache" --progress-output "$full_progress" \
    --progress-interval 25 --prediction-cache "$full_predictions" \
    --output "$full_evaluation"

  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
    --baseline "$full_baseline" --candidate "$full_evaluation" \
    --minimum-relative-improvement 0.0 --output "$full_comparison"
  write_promotion "$full_baseline" "$full_evaluation" "$full_comparison" "$full_promotion" full682

  if "$python_bin" - "$full_promotion" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("promote") else 1)
PY
  then
    "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
      --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
      --checkpoint-dir "$output/best" --split local-test --samples 667 \
      --high-motion-threshold 0.04 --accept-mae 1.0 --failure-gif-count 3 \
      --resume-cache-dir "$local_cache" --progress-output "$local_progress" \
      --progress-interval 25 --prediction-cache "$local_predictions" \
      --output "$local_evaluation"
  else
    echo "Horizon candidate did not pass the full-validation promotion gate; skipping local test."
  fi
else
  echo "Horizon pilot did not pass the preview promotion gate; skipping full evaluation."
fi
