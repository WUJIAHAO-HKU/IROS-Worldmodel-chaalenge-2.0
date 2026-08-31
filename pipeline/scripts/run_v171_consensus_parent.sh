#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_root="artifacts/strict_track2_joint_augmentation_20260810/v171_consensus_action_recursive8"
resume_args=()
if [[ -f "$run_root/training_state.pt" ]]; then
  resume_args=(--resume)
elif [[ -e "$run_root" ]]; then
  echo "refusing non-resumable existing output: $run_root" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/pipeline${PYTHONPATH:+:$PYTHONPATH}"
exec /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
  --windows artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4 \
  --split-manifest artifacts/strict_track2_joint_augmentation_20260810/onpolicy_windows_full128_stride4/split_manifest.json \
  --output "$run_root" \
  --init-autoregressive-checkpoint artifacts/strict_track2_joint_augmentation_20260810/v155_joint_domain_parent_3000/best \
  --normalization-checkpoint artifacts/releases/track2_v15_best/v8/baseline/autoregressive \
  --statistics-cache artifacts/strict_track2_joint_augmentation_20260810/v170_right_arm_training_stats.npz \
  --steps 400 \
  --batch-size 1 \
  --learning-rate 0.000005 \
  --train-rollout-steps 8 \
  --motion-weight 3 \
  --motion-threshold 0.02 \
  --horizon-loss-power 1.5 \
  --temporal-delta-weight 1 \
  --texture-laplacian-weight 0.5 \
  --lowfreq-anchor-weight 0.25 \
  --high-motion-threshold 0.04 \
  --high-motion-oversample-factor 3 \
  --right-arm-oversample-factor 4 \
  --arm-sampling-source consensus \
  --arm-dominance-margin 1.10 \
  --high-motion-selection-weight 0.25 \
  --right-action-selection-weight 0.50 \
  --validation-interval 100 \
  --validation-batches 64 \
  --checkpoint-interval 50 \
  --seed 20260814 \
  --num-workers 2 \
  "${resume_args[@]}"
