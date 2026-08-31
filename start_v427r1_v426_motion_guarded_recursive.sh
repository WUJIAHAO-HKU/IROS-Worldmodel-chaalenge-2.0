#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_official_20260810/run_registry/v427r1_v426_motion_guarded_recursive_seed1577_20260823"
mkdir -p "$run"
export PYTHONPATH="pipeline:pipeline/scripts"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
if [[ ! -f "$run/preregistration.json" ]]; then
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v427r1_v426_motion_guarded_recursive.py
fi
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/audit_v427r1_v426_motion_guarded_recursive.py \
  --baseline-checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v209_v202_v208_public_arm_routed_release \
  --candidate-checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v426_v423_motion_guarded_right_release \
  --windows artifacts/adjust_bottle_windows_full \
  --instruction-map artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
  --reward-checkpoint artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt \
  --t5-model artifacts/official_resources/reward_model/t5-base \
  --preregistration "$run/preregistration.json" --output "$run/recursive_gate_report.json" \
  --device cuda --inference-batch-size 8 --reward-batch-size 32
