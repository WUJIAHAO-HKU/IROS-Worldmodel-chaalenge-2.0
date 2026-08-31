#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
WINDOWS="$JOINT/pilot_joint_parent_windows"
OUTPUT="$JOINT/pilot_parent_bounded_residual_300"
BASE="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"
POSE="$JOINT/reconstructed_pose_v170/best.pt"

test -f "$WINDOWS/split_manifest.json"
test -f "$BASE/model.pt"
test -f "$BASE/action_normalization.npz"
test -f "$POSE"
test ! -e "$OUTPUT/latest/parent_extension.pt"

cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON_BIN" pipeline/scripts/train_action_occlusion_texture_parent_v230.py \
  --windows "$WINDOWS" \
  --split-manifest "$WINDOWS/split_manifest.json" \
  --init-checkpoint "$BASE" \
  --pose-checkpoint "$POSE" \
  --output "$OUTPUT" \
  --steps 300 \
  --batch-size 1 \
  --learning-rate 2e-4 \
  --validation-interval 25 \
  --validation-windows 32 \
  --state-strength 0 \
  --temporal-weight 1.25 \
  --texture-weight 2 \
  --contact-weight 0.5 \
  --protect-weight 5 \
  --maximum-flow-pixels 0 \
  --maximum-structure-residual 0.15 \
  --maximum-texture-residual 0.08 \
  --initial-gate-logit -2 \
  --data-boundary "supplied demos plus declared public RoboTwin Pi0.5 train-only synthetic trajectories" \
  --evaluation-residual-ema 0.6 \
  --seed 20260810
