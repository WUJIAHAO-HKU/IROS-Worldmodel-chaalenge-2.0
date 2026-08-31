#!/usr/bin/env bash
set -euo pipefail
R='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v333_v326_v325_hybrid_onset_terminal_gate_seed1503_20260822"
PHASE="$J/v323_public_terminal_phase_gate_seed1493_20260822"
COMMON=(
  --checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release"
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz"
  --phase-gate "$PHASE/terminal_phase_gate.npz"
  --windows "$R/artifacts/adjust_bottle_windows_full"
  --device cuda
)
cd "$R"
export PYTHONPATH="$R/pipeline:$R/pipeline/scripts"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
PY='/root/miniconda3/envs/go1/bin/python'
taskset -c 0-7 "$PY" pipeline/scripts/test_v333_hybrid_onset_terminal.py \
  "${COMMON[@]}" --output "$RUN/audit/contract_report.json" >>"$RUN/audit/contract.log" 2>&1
taskset -c 0-7 "$PY" pipeline/scripts/audit_v333_hybrid_onset_terminal_gate.py \
  "${COMMON[@]}" --split-manifest "$R/artifacts/splits/adjust_bottle_50episodes_full.json" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$R/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$R/artifacts/official_resources/reward_model/t5-base" \
  --contract-report "$RUN/audit/contract_report.json" \
  --training-report "$J/v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json" \
  --phase-training-report "$PHASE/audit/training_report.json" \
  --output "$RUN/audit/causal_gate_report.json" --inference-batch-size 8 --reward-batch-size 32 \
  >>"$RUN/audit/causal_gate.log" 2>&1
taskset -c 0-7 "$PY" pipeline/scripts/audit_v333_public_recursive_stability.py \
  "${COMMON[@]}" --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$R/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$R/artifacts/official_resources/reward_model/t5-base" \
  --contract-report "$RUN/audit/contract_report.json" --causal-report "$RUN/audit/causal_gate_report.json" \
  --output "$RUN/audit/recursive_stability_report.json" --reward-batch-size 32 \
  >>"$RUN/audit/recursive_stability.log" 2>&1
taskset -c 0-1 "$PY" pipeline/scripts/authorize_expensive_rl.py \
  --candidate-manifest "$RUN/release_registration.json" \
  --absolute-gate "contract=$RUN/audit/contract_report.json" \
  --absolute-gate "action_gate_training=$J/v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json" \
  --absolute-gate "terminal_phase_training=$PHASE/audit/training_report.json" \
  --absolute-gate "recursive_stability=$RUN/audit/recursive_stability_report.json" \
  --causal-gate "$RUN/audit/causal_gate_report.json" \
  --output "$RUN/audit/expensive_rl_authorization.json" >>"$RUN/audit/authorization.log" 2>&1
printf 'V333_HYBRID_ONSET_TERMINAL_GATE_PASSED\n'
