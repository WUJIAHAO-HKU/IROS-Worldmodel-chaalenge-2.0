#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
O="$ROOT/artifacts/strict_track2_official_20260810"
RUN="$J/v417_reward_monotone_successor_seed1571_20260823"
REG="$O/run_registry/v417_reward_monotone_successor_seed1571_20260823"
PY='/root/miniconda3/envs/go1/bin/python'
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
REWARD_MAP="$J/v243b_public_clean_successor_reward_map_seed1445_20260819/audit/clean_successor_rewards.npz"

test -s "$RUN/release_registration.json"
test ! -e "$RUN/RECURSIVE_COMPLETE"
cleanup() {
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start \
    >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export WAM_V312_ACTION_GATE="$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz"
export WAM_V324_PHASE_GATE="$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz"
export WAM_V417_REWARD_MAP="$REWARD_MAP"

set +e
taskset -c 0-11 "$PY" "$ROOT/pipeline/scripts/audit_v417_recursive_reward_causal.py" \
  --baseline-checkpoint-dir "$J/v209_v202_v208_public_arm_routed_release" \
  --candidate-checkpoint-dir "$RUN/release" \
  --library-index "$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --instruction-map "$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --preregistration "$RUN/release_registration.json" \
  --output "$RUN/audit/recursive_reward_causal.json" --device cuda \
  --inference-batch-size 8 --reward-batch-size 32 \
  >"$RUN/audit/recursive_reward_causal.log" 2>&1
status=$?
set -e
if test -s "$RUN/audit/recursive_reward_causal.json"; then
  cp "$RUN/audit/recursive_reward_causal.json" "$REG/recursive_reward_causal.json"
fi
touch "$RUN/RECURSIVE_COMPLETE"
printf '%s\n' "$status" >"$REG/audit_exit_code.txt"
echo "V417_RECURSIVE_AUDIT_EXIT=$status"
exit "$status"
