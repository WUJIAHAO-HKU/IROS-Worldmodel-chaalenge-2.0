#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v351_bridge_focused_train_seed1520_20260822"
fit="artifacts/strict_track2_joint_augmentation_20260810/v350_public_residual_bridge_fit_seed1519_20260822"
v339="artifacts/strict_track2_joint_augmentation_20260810/v339_high_specificity_recursive_ood_gate_seed1508_20260822"
v340="artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822"
mkdir -p "$run"; export PYTHONPATH="pipeline:pipeline/scripts"; export CUDA_VISIBLE_DEVICES=0
if [[ ! -f "$run/release_registration.json" ]]; then /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v351_bridge_focused_train.py; fi
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python pipeline/scripts/audit_v351_bridge_focused_train.py \
  --checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v209_v202_v208_public_arm_routed_release \
  --library-index artifacts/strict_track2_joint_augmentation_20260810/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz \
  --action-gate artifacts/strict_track2_joint_augmentation_20260810/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz \
  --phase-gate artifacts/strict_track2_joint_augmentation_20260810/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz \
  --recursive-ood-gate "$v339/recursive_ood_gate.npz" --bridge-profile "$fit/residual_bridge_profile.npz" \
  --windows artifacts/adjust_bottle_windows_full \
  --instruction-map artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
  --reward-checkpoint artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt \
  --t5-model artifacts/official_resources/reward_model/t5-base \
  --v340-train-report "$v340/train_recursive_gate_report.json" --v350-fit-report "$fit/fit_report.json" \
  --preregistration "$run/release_registration.json" --output "$run/focused_train_gate_report.json" \
  --device cuda --inference-batch-size 8 --reward-batch-size 32 --feature-workers 8
