#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v392_v164s50_vs_v209_parametric_recursive_seed1553_20260823"
PY=/root/miniconda3/envs/go1/bin/python
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
taskset -c 0-15 "$PY" "$ROOT/pipeline/scripts/audit_v392_v164s50_parametric_recursive.py" \
  --baseline-checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release" \
  --candidate-checkpoint-dir "$RUN/release" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --preregistration "$RUN/release_registration.json" \
  --output "$RUN/recursive_gate_report.json" \
  --device cuda --inference-batch-size 8 --reward-batch-size 32 \
  >"$RUN/recursive_gate.log" 2>&1
touch "$RUN/COMPLETE"
echo V392_PARAMETRIC_RECURSIVE_COMPLETE
