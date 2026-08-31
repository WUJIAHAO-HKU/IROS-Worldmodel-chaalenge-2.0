#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v366_serial_consistent_v355_v326_seed1530_20260822"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts"
exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python "$ROOT/pipeline/scripts/test_v366_serial_consistent_v355_v326.py" \
  --checkpoint-dir "$J/v355_v202_v354_parametric_arm_routed_release" \
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --action-gate "$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz" \
  --phase-gate "$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --output "$RUN/contract_report.json" \
  --device cuda
