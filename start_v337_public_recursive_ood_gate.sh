#!/usr/bin/env bash
set -euo pipefail

cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v337_public_recursive_ood_gate_seed1507_20260822"
mkdir -p "$run"
export PYTHONPATH="pipeline:pipeline/scripts"
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v337_public_recursive_ood_gate.py
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/train_v337_public_recursive_ood_gate.py \
  --checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v209_v202_v208_public_arm_routed_release \
  --library-index artifacts/strict_track2_joint_augmentation_20260810/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz \
  --action-gate artifacts/strict_track2_joint_augmentation_20260810/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz \
  --phase-gate artifacts/strict_track2_joint_augmentation_20260810/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz \
  --windows artifacts/adjust_bottle_windows_full \
  --split artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
  --preregistration "$run/release_registration.json" \
  --output "$run/recursive_ood_gate.npz" \
  --report "$run/training_report.json" \
  --device cuda \
  --batch-size 8
