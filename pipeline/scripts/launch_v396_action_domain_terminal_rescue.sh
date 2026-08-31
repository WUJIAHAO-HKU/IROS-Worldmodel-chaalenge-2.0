#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v396_action_domain_terminal_rescue_seed1556_20260823"
PY=/root/miniconda3/envs/go1/bin/python
export PYTHONPATH="$ROOT/pipeline"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
test -s "$RUN/preregistration.json"
test ! -e "$RUN/training_report.json"
taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/train_v396_action_domain_terminal_rescue.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --onpolicy-windows "$J/onpolicy_windows_full128_stride4" \
  --onpolicy-split "$J/onpolicy_windows_full128_stride4/split_manifest.json" \
  --phase-gate "$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz" \
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz" \
  --preregistration "$RUN/preregistration.json" \
  --output "$RUN/action_phase_gate.npz" \
  --report "$RUN/training_report.json" >"$RUN/training.log" 2>&1
touch "$RUN/TRAINING_COMPLETE"
echo V396_ACTION_DOMAIN_TERMINAL_RESCUE_COMPLETE
