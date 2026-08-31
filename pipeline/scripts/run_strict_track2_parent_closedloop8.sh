#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
WINDOWS="$JOINT/pilot_joint_parent_windows"
OUTPUT="$JOINT/pilot_parent_closedloop8_200"
BASE="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"

test -f "$WINDOWS/split_manifest.json"
test -f "$BASE/model.pt"
test -f "$BASE/action_normalization.npz"
if [[ -e "$OUTPUT/training_state.pt" ]]; then
  RESUME=(--resume)
else
  RESUME=()
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON_BIN" pipeline/scripts/train_autoregressive_unet.py \
  --windows "$WINDOWS" \
  --split-manifest "$WINDOWS/split_manifest.json" \
  --output "$OUTPUT" \
  --init-autoregressive-checkpoint "$BASE" \
  --normalization-checkpoint "$BASE" \
  --steps 200 \
  --batch-size 1 \
  --learning-rate 2e-6 \
  --train-rollout-steps 8 \
  --motion-weight 3 \
  --motion-threshold 0.02 \
  --horizon-loss-power 0.5 \
  --temporal-delta-weight 1 \
  --texture-laplacian-weight 0.5 \
  --high-motion-threshold 0.04 \
  --high-motion-oversample-factor 3 \
  --high-motion-selection-weight 0.5 \
  --validation-interval 50 \
  --validation-batches 16 \
  --checkpoint-interval 50 \
  --statistics-cache "$JOINT/pilot_joint_training_stats.npz" \
  --seed 20260810 \
  "${RESUME[@]}"
