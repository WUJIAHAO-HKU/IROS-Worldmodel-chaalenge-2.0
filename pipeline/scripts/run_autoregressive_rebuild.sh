#!/usr/bin/env bash
# Rebuild the deleted historical autoregressive baseline with resume-safe stages.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/go1/bin/python}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
one_step="${WAM_AUTOREG_ONE_STEP_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-native256-rebuild-v1}"
rollout="${WAM_AUTOREG_ROLLOUT_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-rebuild-v1}"
candidate="${WAM_AUTOREG_REBUILD_CANDIDATE:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_preview64.json}"
baseline="${WAM_AUTOREG_REBUILD_BASELINE:-artifacts/evaluations/direct_flow_formal_v2_preview64.json}"
comparison="${WAM_AUTOREG_REBUILD_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_comparison.json}"
full_evaluation="${WAM_AUTOREG_REBUILD_FULL_EVALUATION:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_validation_all.json}"
full_cache="${WAM_AUTOREG_REBUILD_FULL_CACHE:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_validation_cache}"
full_progress="${WAM_AUTOREG_REBUILD_FULL_PROGRESS:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_validation_progress.json}"
full_predictions="${WAM_AUTOREG_REBUILD_FULL_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_rebuild_validation_predictions.npz}"
statistics="${WAM_AUTOREG_STATISTICS_CACHE:-artifacts/cache/autoregressive_adjust_bottle_train_statistics.npz}"

# Historical stage 1 used a now-deleted residual initializer. Training from
# scratch keeps the data/architecture contract identical and is the only
# reproducible replacement that does not fabricate incompatible weights.
"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows "$windows" --split-manifest "$split" --output "$one_step" \
  --steps 8000 --batch-size 4 --learning-rate 5e-5 --train-rollout-steps 1 \
  --validation-interval 500 --validation-batches 32 --checkpoint-interval 100 \
  --statistics-cache "$statistics" --seed 20260804 --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows "$windows" --split-manifest "$split" --output "$rollout" \
  --init-autoregressive-checkpoint "$one_step/best" \
  --steps 5000 --batch-size 2 --learning-rate 1e-5 --train-rollout-steps 8 \
  --validation-interval 500 --validation-batches 32 --checkpoint-interval 100 \
  --statistics-cache "$statistics" --seed 20260805 --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$rollout/best" --split validation --samples 64 \
  --failure-gif-count 0 --output "$candidate"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --minimum-relative-improvement 0.01 \
  --output "$comparison"

if "$python_bin" - "$comparison" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    comparison = json.load(handle)
raise SystemExit(0 if comparison.get("promote_to_full_682_evaluation") else 1)
PY
then
  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend autoregressive-unet --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$rollout/best" --split validation --samples 682 \
    --high-motion-threshold 0.04 --accept-mae 1.0 --failure-gif-count 3 \
    --resume-cache-dir "$full_cache" --progress-output "$full_progress" \
    --progress-interval 25 --prediction-cache "$full_predictions" \
    --output "$full_evaluation"
else
  echo "Autoregressive rebuild did not meet the pilot promotion criterion; skipping full evaluation."
fi
