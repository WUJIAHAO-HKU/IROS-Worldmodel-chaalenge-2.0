#!/usr/bin/env bash
set -euo pipefail

cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v336_recursive_context_ood_diagnostic_seed1506"
mkdir -p "$run"
export PYTHONPATH=pipeline
export CUDA_VISIBLE_DEVICES=0
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/diagnose_v336_recursive_context_ood.py \
  --checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v209_v202_v208_public_arm_routed_release \
  --library-index artifacts/strict_track2_joint_augmentation_20260810/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz \
  --action-gate artifacts/strict_track2_joint_augmentation_20260810/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz \
  --phase-gate artifacts/strict_track2_joint_augmentation_20260810/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz \
  --windows artifacts/adjust_bottle_windows_full \
  --device cuda \
  --batch-size 8 \
  --output "$run/report.json"
