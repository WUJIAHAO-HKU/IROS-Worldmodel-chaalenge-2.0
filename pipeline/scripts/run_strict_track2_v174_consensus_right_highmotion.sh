#!/usr/bin/env bash
# Public-data-only parent-world-model specialization.  This never reads any
# evaluation rollout or outcome; model selection uses the public split only.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_root="artifacts/strict_track2_joint_augmentation_20260810/v174_v173_public_consensus_right_highmotion"
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
  --init-autoregressive-checkpoint artifacts/strict_track2_joint_augmentation_20260810/v173_v172_public_long16_terminal_focus/best \
  --normalization-checkpoint artifacts/releases/track2_v15_best/v8/baseline/autoregressive \
  --statistics-cache artifacts/strict_track2_joint_augmentation_20260810/v174_public_consensus_right_highmotion_stats.npz \
  --steps 500 \
  --batch-size 4 \
  --learning-rate 0.000005 \
  --rollout-horizon 16 \
  --train-rollout-steps 16 \
  --motion-weight 3 \
  --motion-threshold 0.02 \
  --horizon-loss-power 2.5 \
  --temporal-delta-weight 1 \
  --texture-laplacian-weight 0.5 \
  --high-motion-threshold 0.04 \
  --high-motion-oversample-factor 5 \
  --right-arm-oversample-factor 8 \
  --arm-sampling-source consensus \
  --arm-dominance-margin 1.10 \
  --high-motion-selection-weight 0.50 \
  --right-action-selection-weight 0.25 \
  --right-high-motion-selection-weight 0.50 \
  --validation-interval 50 \
  --validation-batches 64 \
  --checkpoint-interval 50 \
  --seed 1257 \
  --num-workers 2 \
  "${resume_args[@]}"
