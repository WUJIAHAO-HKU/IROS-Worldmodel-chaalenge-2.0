#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"; run="artifacts/strict_track2_joint_augmentation_20260810/v359_v358_vs_v355_recursive_seed1526_20260822"; mkdir -p "$run"; export PYTHONPATH="pipeline:pipeline/scripts"; export CUDA_VISIBLE_DEVICES=0
if [[ ! -f "$run/release_registration.json" ]]; then /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v359_v358_vs_v355_recursive.py; fi
exec taskset -c 0-7 /root/miniconda3/envs/go1/bin/python pipeline/scripts/audit_v359_v358_vs_v355_recursive.py \
 --baseline-checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v355_v202_v354_parametric_arm_routed_release \
 --candidate-checkpoint-dir artifacts/strict_track2_joint_augmentation_20260810/v358_v202_v357_temporal_arm_routed_release \
 --windows artifacts/adjust_bottle_windows_full --instruction-map artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json \
 --reward-checkpoint artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt --t5-model artifacts/official_resources/reward_model/t5-base \
 --preregistration "$run/release_registration.json" --output "$run/recursive_gate_report.json" --device cuda --inference-batch-size 8 --reward-batch-size 32
