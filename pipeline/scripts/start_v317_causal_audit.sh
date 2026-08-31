#!/usr/bin/env bash
set -uo pipefail
R='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v317_v315_native_batch_causal_gate_seed1490_20260822"
cd "$R"
exec taskset -c 0-7 env \
  PYTHONPATH="$R/pipeline" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
  /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/audit_v317_batched_sparse_failure_terminal_gate.py \
  --checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release" \
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz" \
  --windows "$R/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$R/artifacts/splits/adjust_bottle_50episodes_full.json" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$R/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$R/artifacts/official_resources/reward_model/t5-base" \
  --contract-report "$RUN/audit/contract_report.json" \
  --training-report "$J/v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json" \
  --output "$RUN/audit/causal_gate_report.json" \
  --device cuda --inference-batch-size 8 --reward-batch-size 32 \
  >>"$RUN/audit/causal_gate.log" 2>&1
