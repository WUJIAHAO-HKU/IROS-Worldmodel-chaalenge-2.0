#!/usr/bin/env bash
# Public-data-only Track 2 parent-world-model refinement.  This run is
# outcome-free: it reads only the declared public 112/16 episode split and
# chooses checkpoints solely by reconstructed future-observation metrics.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_root="artifacts/strict_track2_joint_augmentation_20260810/v178_v177_public_right_stable_refine"
if [[ -e "$run_root" ]]; then
  echo "refusing existing output: $run_root" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/pipeline${PYTHONPATH:+:$PYTHONPATH}"
exec /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
  --windows artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4 \
  --split-manifest artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4/split_manifest.json \
  --output "$run_root" \
  --init-autoregressive-checkpoint artifacts/strict_track2_joint_augmentation_20260810/v177_v175_v176_public_balanced_blend_alpha050 \
  --normalization-checkpoint artifacts/releases/track2_v15_best/v8/baseline/autoregressive \
  --statistics-cache artifacts/strict_track2_joint_augmentation_20260810/v174_public_consensus_right_highmotion_stats.npz \
  --steps 250 \
  --batch-size 4 \
  --learning-rate 0.0000005 \
  --rollout-horizon 16 \
  --train-rollout-steps 16 \
  --motion-weight 4 \
  --motion-threshold 0.025 \
  --horizon-loss-power 3 \
  --temporal-delta-weight 1.5 \
  --texture-laplacian-weight 0.5 \
  --high-motion-threshold 0.03 \
  --high-motion-oversample-factor 12 \
  --right-arm-oversample-factor 24 \
  --arm-sampling-source consensus \
  --arm-dominance-margin 1.10 \
  --high-motion-selection-weight 0.25 \
  --right-action-selection-weight 0.35 \
  --right-high-motion-selection-weight 1.0 \
  --validation-interval 25 \
  --validation-batches 166 \
  --checkpoint-interval 25 \
  --seed 1279 \
  --num-workers 2
