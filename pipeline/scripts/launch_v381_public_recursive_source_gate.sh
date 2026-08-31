#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";RUN="$J/v381_public_recursive_source_gate_seed1544_20260823";PY=/root/miniconda3/envs/go1/bin/python
export PYTHONPATH="$ROOT/pipeline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
test -s "$RUN/preregistration.json";test ! -e "$RUN/training_report.json"
taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/train_v381_public_recursive_source_gate.py" --v355-release "$J/v355_v202_v354_parametric_arm_routed_release" --v378-release "$J/v378_source_routed_blended_cartesian_seed1541_20260823/release" --library "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" --windows "$ROOT/artifacts/adjust_bottle_windows_full" --split "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" --preregistration "$RUN/preregistration.json" --output "$RUN/recursive_source_gate.npz" --report "$RUN/training_report.json" --device cuda --batch-size 8 >"$RUN/training.log" 2>&1
touch "$RUN/TRAINING_COMPLETE";printf 'V381_SOURCE_GATE_COMPLETE\n'
