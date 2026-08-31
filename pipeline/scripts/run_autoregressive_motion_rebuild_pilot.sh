#!/usr/bin/env bash
# Test whether motion-aware fine-tuning improves the rebuilt autoregressive model.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/go1/bin/python}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
parent="${WAM_AUTOREG_PARENT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-rebuild-v1/best}"
output="${WAM_AUTOREG_MOTION_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-motion-rebuild-pilot-v1}"
baseline="${WAM_AUTOREG_MOTION_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_preview64.json}"
candidate="${WAM_AUTOREG_MOTION_CANDIDATE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_pilot_preview64.json}"
predictions="${WAM_AUTOREG_MOTION_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_pilot_preview64_predictions.npz}"
comparison="${WAM_AUTOREG_MOTION_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_pilot_comparison.json}"
promotion="${WAM_AUTOREG_MOTION_PROMOTION:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_pilot_promotion.json}"
full_evaluation="${WAM_AUTOREG_MOTION_FULL_EVALUATION:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_validation_all.json}"
full_baseline="${WAM_AUTOREG_MOTION_FULL_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_validation_all.json}"
full_comparison="${WAM_AUTOREG_MOTION_FULL_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_vs_parent_full682.json}"
full_cache="${WAM_AUTOREG_MOTION_FULL_CACHE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_validation_cache}"
full_progress="${WAM_AUTOREG_MOTION_FULL_PROGRESS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_validation_progress.json}"
full_predictions="${WAM_AUTOREG_MOTION_FULL_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_validation_predictions.npz}"
local_evaluation="${WAM_AUTOREG_MOTION_LOCAL_EVALUATION:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_local_test_all.json}"
local_cache="${WAM_AUTOREG_MOTION_LOCAL_CACHE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_local_test_cache}"
local_progress="${WAM_AUTOREG_MOTION_LOCAL_PROGRESS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_local_test_progress.json}"
local_predictions="${WAM_AUTOREG_MOTION_LOCAL_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_rebuild_local_test_predictions.npz}"
statistics="${WAM_AUTOREG_STATISTICS_CACHE:-artifacts/cache/autoregressive_adjust_bottle_train_statistics.npz}"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows "$windows" --split-manifest "$split" --output "$output" \
  --init-autoregressive-checkpoint "$parent" \
  --steps 2000 --batch-size 2 --learning-rate 5e-6 --train-rollout-steps 8 \
  --motion-weight 2.5 --motion-threshold 0.03 \
  --high-motion-threshold 0.04 --high-motion-oversample-factor 3.0 \
  --high-motion-selection-weight 0.5 \
  --validation-interval 250 --validation-batches 64 --checkpoint-interval 100 \
  --statistics-cache "$statistics" --seed 20260808 --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$output/best" --split validation --samples 64 \
  --high-motion-threshold 0.04 --failure-gif-count 0 \
  --prediction-cache "$predictions" --output "$candidate"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --minimum-relative-improvement 0.0 \
  --output "$comparison"

"$python_bin" - "$baseline" "$candidate" "$comparison" "$promotion" <<'PY'
import json
import os
import sys
from pathlib import Path

baseline_path, candidate_path, comparison_path, output_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text())
candidate = json.loads(candidate_path.read_text())
comparison = json.loads(comparison_path.read_text())
baseline_mae = float(baseline["model_mae_mean"])
candidate_mae = float(candidate["model_mae_mean"])
baseline_high = float(baseline["high_motion_model_mae_mean"])
candidate_high = float(candidate["high_motion_model_mae_mean"])
overall_relative_improvement = (baseline_mae - candidate_mae) / baseline_mae
high_motion_relative_improvement = (baseline_high - candidate_high) / baseline_high
promote = overall_relative_improvement >= -0.002 and high_motion_relative_improvement >= 0.01
result = {
    "format": "track2-autoregressive-motion-pilot-promotion-v1",
    "baseline": str(baseline_path.resolve()),
    "candidate": str(candidate_path.resolve()),
    "comparison": str(comparison_path.resolve()),
    "maximum_overall_relative_regression": 0.002,
    "minimum_high_motion_relative_improvement": 0.01,
    "overall_relative_improvement": overall_relative_improvement,
    "high_motion_relative_improvement": high_motion_relative_improvement,
    "paired_bootstrap_mean_mae_delta_ci95": comparison.get("paired_bootstrap_mean_mae_delta_ci95"),
    "promote_to_full_682_evaluation": promote,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, output_path)
print(json.dumps(result, indent=2))
PY

if "$python_bin" - "$promotion" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    promotion = json.load(handle)
raise SystemExit(0 if promotion.get("promote_to_full_682_evaluation") else 1)
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
    --minimum-relative-improvement 0.01 --output "$full_comparison"

  if "$python_bin" - "$full_comparison" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    comparison = json.load(handle)
raise SystemExit(0 if comparison.get("promote_to_full_682_evaluation") else 1)
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
    echo "Motion candidate did not improve full validation by 1%; skipping local-test evaluation."
  fi
else
  echo "Motion pilot did not meet the promotion criterion; skipping full evaluation."
fi
