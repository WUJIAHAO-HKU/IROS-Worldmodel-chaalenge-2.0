#!/usr/bin/env bash
# Reproducible full validation/local-test evaluation of the promoted horizon-aware gate.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
checkpoint="${WAM_AUTOREG_HORIZON_GATE_CHECKPOINT:-artifacts/checkpoints/autoregressive-horizon-direct-flow-motion-gated-v2}"
validation="${WAM_AUTOREG_HORIZON_GATE_VALIDATION:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_validation_all.json}"
validation_cache="${WAM_AUTOREG_HORIZON_GATE_VALIDATION_CACHE:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_validation_cache}"
validation_progress="${WAM_AUTOREG_HORIZON_GATE_VALIDATION_PROGRESS:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_validation_progress.json}"
validation_predictions="${WAM_AUTOREG_HORIZON_GATE_VALIDATION_PREDICTIONS:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_validation_predictions.npz}"
validation_comparison="${WAM_AUTOREG_HORIZON_GATE_VALIDATION_COMPARISON:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_vs_single_full682.json}"
validation_baseline="${WAM_AUTOREG_HORIZON_GATE_VALIDATION_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_validation_all.json}"
local="${WAM_AUTOREG_HORIZON_GATE_LOCAL:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_local_test_all.json}"
local_cache="${WAM_AUTOREG_HORIZON_GATE_LOCAL_CACHE:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_local_test_cache}"
local_progress="${WAM_AUTOREG_HORIZON_GATE_LOCAL_PROGRESS:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_local_test_progress.json}"
local_predictions="${WAM_AUTOREG_HORIZON_GATE_LOCAL_PREDICTIONS:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_local_test_predictions.npz}"
local_comparison="${WAM_AUTOREG_HORIZON_GATE_LOCAL_COMPARISON:-artifacts/evaluations/autoregressive_horizon_direct_flow_motion_gate_vs_single_local667.json}"
local_baseline="${WAM_AUTOREG_HORIZON_GATE_LOCAL_BASELINE:-artifacts/evaluations/autoregressive_unet_rollout8_horizon_local_test_all.json}"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-flow-ensemble --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$checkpoint" --split validation --samples 682 \
  --high-motion-threshold 0.04 --accept-mae 1.0 --failure-gif-count 0 \
  --resume-cache-dir "$validation_cache" --progress-output "$validation_progress" \
  --progress-interval 25 --prediction-cache "$validation_predictions" --output "$validation"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$validation_baseline" \
  --candidate "$validation" --minimum-relative-improvement 0.005 --output "$validation_comparison"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-flow-ensemble --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$checkpoint" --split local-test --samples 667 \
  --high-motion-threshold 0.04 --accept-mae 1.0 --failure-gif-count 0 \
  --resume-cache-dir "$local_cache" --progress-output "$local_progress" \
  --progress-interval 25 --prediction-cache "$local_predictions" --output "$local"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$local_baseline" \
  --candidate "$local" --minimum-relative-improvement 0.0 --output "$local_comparison"
