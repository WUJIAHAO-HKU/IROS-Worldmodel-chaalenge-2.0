#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";RUN="$J/v380_terminal_hold_diagnostic_seed1543_20260823";PY=/root/miniconda3/envs/go1/bin/python
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
test -s "$RUN/preregistration.json";test ! -e "$RUN/audit/terminal_hold.json"
taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/audit_v380_terminal_hold_diagnostic.py" --baseline-release "$J/v355_v202_v354_parametric_arm_routed_release" --v375-release "$J/v375_bounded_cartesian_phase_pilot_seed1538_20260823/release" --library "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" --gate "$J/v339_high_specificity_recursive_ood_gate_seed1508_20260822/recursive_ood_gate.npz" --windows "$ROOT/artifacts/adjust_bottle_windows_full" --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" --preregistration "$RUN/preregistration.json" --output "$RUN/audit/terminal_hold.json" --device cuda --batch-size 8 >"$RUN/audit/terminal_hold.log" 2>&1
touch "$RUN/TERMINAL_HOLD_COMPLETE";printf 'V380_TERMINAL_HOLD_COMPLETE\n'
