#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v393_public_failure_calibrated_phase_seed1554_20260823"
PY=/root/miniconda3/envs/go1/bin/python
export PYTHONPATH="$ROOT/pipeline"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
test -s "$RUN/preregistration.json"
test ! -e "$RUN/training_report.json"
taskset -c 0-15 "$PY" "$ROOT/pipeline/scripts/train_v393_failure_calibrated_phase_classifier.py" \
  --v326-release "$J/v209_v202_v208_public_arm_routed_release" \
  --v355-release "$J/v355_v202_v354_parametric_arm_routed_release" \
  --v378-release "$J/v378_source_routed_blended_cartesian_seed1541_20260823/release" \
  --library "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --onpolicy-windows "$J/onpolicy_windows_full128_stride4" \
  --onpolicy-split "$J/onpolicy_windows_full128_stride4/split_manifest.json" \
  --phase-gate "$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz" \
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz" \
  --preregistration "$RUN/preregistration.json" \
  --output "$RUN/failure_calibrated_phase_gate.npz" \
  --report "$RUN/training_report.json" \
  --device cuda --batch-size 8 >"$RUN/training.log" 2>&1
touch "$RUN/TRAINING_COMPLETE"
echo V393_FAILURE_CALIBRATED_PHASE_COMPLETE
