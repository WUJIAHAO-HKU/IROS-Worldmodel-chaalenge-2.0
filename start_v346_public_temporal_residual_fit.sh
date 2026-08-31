#!/usr/bin/env bash
set -euo pipefail

cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v346_public_temporal_residual_fit_seed1515_20260822"
v339="artifacts/strict_track2_joint_augmentation_20260810/v339_high_specificity_recursive_ood_gate_seed1508_20260822"
mkdir -p "$run"
export PYTHONPATH="pipeline:pipeline/scripts"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
if [[ ! -f "$run/release_registration.json" ]]; then
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v346_public_temporal_residual_fit.py
fi
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/fit_v346_public_temporal_residual_profile.py \
  --checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v209_v202_v208_public_arm_routed_release \
  --library-index artifacts/strict_track2_joint_augmentation_20260810/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz \
  --action-gate artifacts/strict_track2_joint_augmentation_20260810/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz \
  --phase-gate artifacts/strict_track2_joint_augmentation_20260810/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz \
  --recursive-ood-gate "$v339/recursive_ood_gate.npz" \
  --windows artifacts/adjust_bottle_windows_full \
  --instruction-map artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
  --preregistration "$run/release_registration.json" \
  --output-profile "$run/temporal_residual_profile.npz" \
  --output-report "$run/fit_report.json" \
  --device cuda --inference-batch-size 8 --feature-workers 8
