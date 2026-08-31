#!/usr/bin/env bash
# Public-data-only Track 2 parent-world-model refinement.  This run uses only
# the declared 112/16 public split and selects checkpoints exclusively from
# reconstructed future observations; success, reward, and held-out seeds are
# never read here.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_root="artifacts/strict_track2_joint_augmentation_20260810/v176_v175_public_right_dynamics_refine"
if [[ -e "$run_root" ]]; then
  echo "refusing existing output: $run_root" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/pipeline${PYTHONPATH:+:$PYTHONPATH}"
exec /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
  --windows artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4 \
  --split-manifest artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4/split_manifest.json \
  --output "$run_root" \
  --init-autoregressive-checkpoint artifacts/strict_track2_joint_augmentation_20260810/v175_v174_public_right_highmotion_refine/best \
  --normalization-checkpoint artifacts/releases/track2_v15_best/v8/baseline/autoregressive \
  --statistics-cache artifacts/strict_track2_joint_augmentation_20260810/v174_public_consensus_right_highmotion_stats.npz \
  --steps 400 \
  --batch-size 4 \
  --learning-rate 0.000001 \
  --rollout-horizon 16 \
  --train-rollout-steps 16 \
  --motion-weight 4 \
  --motion-threshold 0.02 \
  --horizon-loss-power 3 \
  --temporal-delta-weight 1.5 \
  --texture-laplacian-weight 0.5 \
  --high-motion-threshold 0.025 \
  --high-motion-oversample-factor 16 \
  --right-arm-oversample-factor 32 \
  --arm-sampling-source consensus \
  --arm-dominance-margin 1.10 \
  --high-motion-selection-weight 0.25 \
  --right-action-selection-weight 0.25 \
  --right-high-motion-selection-weight 1.0 \
  --validation-interval 50 \
  --validation-batches 166 \
  --checkpoint-interval 50 \
  --seed 1276 \
  --num-workers 2
