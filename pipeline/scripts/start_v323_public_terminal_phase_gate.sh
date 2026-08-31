#!/usr/bin/env bash
set -euo pipefail
R='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v323_public_terminal_phase_gate_seed1493_20260822"
cd "$R"
exec taskset -c 0-7 env \
  PYTHONPATH="$R/pipeline" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/build_v323_public_terminal_phase_gate.py \
  --checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release" \
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$R/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$R/artifacts/official_resources/reward_model/t5-base" \
  --preregistration "$RUN/preregistration.json" \
  --output "$RUN/terminal_phase_gate.npz" \
  --report "$RUN/audit/training_report.json" \
  --device cuda --reward-batch-size 32 \
  >>"$RUN/audit/training.log" 2>&1
