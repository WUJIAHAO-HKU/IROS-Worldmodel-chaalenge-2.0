#!/usr/bin/env bash
# Public-data-only refinement for the Track 2 parent world model.  Model
# selection uses reconstructed future observations on the declared public
# validation split only; it never reads policy success or held-out seeds.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_root="artifacts/strict_track2_joint_augmentation_20260810/v175_v174_public_right_highmotion_refine"
if [[ -e "$run_root" ]]; then
  echo "refusing existing output: $run_root" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/pipeline${PYTHONPATH:+:$PYTHONPATH}"
exec /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
  --windows artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4 \
  --split-manifest artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4/split_manifest.json \
  --output "$run_root" \
  --init-autoregressive-checkpoint artifacts/strict_track2_joint_augmentation_20260810/v174_v173_public_consensus_right_highmotion/best \
  --normalization-checkpoint artifacts/releases/track2_v15_best/v8/baseline/autoregressive \
  --statistics-cache artifacts/strict_track2_joint_augmentation_20260810/v174_public_consensus_right_highmotion_stats.npz \
  --steps 600 \
  --batch-size 4 \
  --learning-rate 0.000002 \
  --rollout-horizon 16 \
  --train-rollout-steps 16 \
  --motion-weight 3 \
  --motion-threshold 0.02 \
  --horizon-loss-power 2.5 \
  --temporal-delta-weight 1 \
  --texture-laplacian-weight 0.5 \
  --high-motion-threshold 0.03 \
  --high-motion-oversample-factor 8 \
  --right-arm-oversample-factor 16 \
  --arm-sampling-source consensus \
  --arm-dominance-margin 1.10 \
  --high-motion-selection-weight 0.25 \
  --right-action-selection-weight 0.25 \
  --right-high-motion-selection-weight 0.75 \
  --validation-interval 50 \
  --validation-batches 166 \
  --checkpoint-interval 50 \
  --seed 1266 \
  --num-workers 2
