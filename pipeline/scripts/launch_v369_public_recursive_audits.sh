#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v369_v354_strong_reward_logit_recursive_pilot_seed1533_20260822"
PY=/root/miniconda3/envs/go1/bin/python
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
test -f "$RUN/TRAINING_COMPLETE"
mkdir -p "$RUN/audit"
for step in 25 50; do
  checkpoint=$(printf '%s/checkpoints/checkpoint_step_%06d' "$RUN" "$step")
  report=$(printf '%s/audit/checkpoint_step_%06d_recursive_gate.json' "$RUN" "$step")
  log=$(printf '%s/audit/checkpoint_step_%06d_recursive_gate.log' "$RUN" "$step")
  set +e
  taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/audit_v369_strong_reward_logit_pilot.py" \
    --candidate-checkpoint "$checkpoint" --checkpoint-step "$step" \
    --baseline-report "$JOINT/v356_v355_parametric_recursive_seed1524_20260822/recursive_gate_report.json" \
    --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
    --instruction-map "$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
    --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
    --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
    --preregistration "$RUN/release_registration.json" --output "$report" \
    --device cuda --inference-batch-size 8 --reward-batch-size 32 >"$log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 && $status -ne 2 ]]; then exit "$status"; fi
done
touch "$RUN/AUDIT_COMPLETE"
printf 'V369_AUDITS_COMPLETE\n'
