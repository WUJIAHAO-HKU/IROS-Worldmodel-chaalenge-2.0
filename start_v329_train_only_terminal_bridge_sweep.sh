#!/usr/bin/env bash
set -euo pipefail
R='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v329_train_only_terminal_bridge_sweep_seed1499_20260822"
cd "$R"
export PYTHONPATH="$R/pipeline:$R/pipeline/scripts"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
PY='/root/miniconda3/envs/go1/bin/python'
taskset -c 0-7 "$PY" pipeline/scripts/sweep_v329_train_only_terminal_bridge.py \
  --checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release" \
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz" \
  --phase-gate "$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz" \
  --windows "$R/artifacts/adjust_bottle_windows_full" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$R/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$R/artifacts/official_resources/reward_model/t5-base" \
  --preregistration "$RUN/preregistration.json" \
  --output "$RUN/audit/sweep_report.json" \
  --device cuda --inference-batch-size 8 --reward-batch-size 32 \
  >>"$RUN/audit/sweep.log" 2>&1
printf 'V329_TRAIN_ONLY_SWEEP_COMPLETED\n'
