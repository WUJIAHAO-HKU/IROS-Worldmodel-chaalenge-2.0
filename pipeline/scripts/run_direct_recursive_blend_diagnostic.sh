#!/usr/bin/env bash
# Cache two low-memory predictors and test their complementarity without retraining.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
direct="${WAM_DIRECT_CHECKPOINT:-artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best}"
recursive="${WAM_RECURSIVE_CHECKPOINT:-artifacts/checkpoints/recursive-flow-unet-track2-pilot-v1/best}"
samples="${WAM_BLEND_SAMPLES:-64}"
root="artifacts/evaluations/direct_recursive_blend_diagnostic"
mkdir -p "$root"

evaluate_cache() {
  local backend=$1 checkpoint=$2 name=$3
  if [[ ! -s "$root/${name}_predictions.npz" ]]; then
    "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
      --backend "$backend" --windows "$windows" --split-manifest "$split" \
      --checkpoint-dir "$checkpoint" --split validation --samples "$samples" \
      --failure-gif-count 0 --progress-interval 8 \
      --resume-cache-dir "$root/${name}_resume" \
      --prediction-cache "$root/${name}_predictions.npz" \
      --output "$root/${name}_report.json"
  fi
}

evaluate_cache direct-flow-unet "$direct" direct
evaluate_cache recursive-flow-unet "$recursive" recursive
"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/sweep_prediction_blend.py \
  --windows "$windows" --split-manifest "$split" --samples "$samples" \
  --first-cache "$root/direct_predictions.npz" --first-name direct-flow \
  --second-cache "$root/recursive_predictions.npz" --second-name recursive-flow \
  --weight-step 0.05 --output "$root/comparison.json"
