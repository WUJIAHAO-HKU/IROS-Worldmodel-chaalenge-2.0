#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405"
P="$BASE/pipeline/scripts"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/miniconda3/envs/go1/bin/python'
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$BASE/pipeline/scripts:$RLINF"

test -f "$RUN/TRAINING_COMPLETE"
test -f "$RUN/audit/baseline/COMPLETE"

for step in 30 60 90 120; do
  tag=$(printf 'step%03d' "$step")
  candidate=$(printf '%s/checkpoints/checkpoint_step_%06d' "$RUN" "$step")
  out="$RUN/audit/$tag"
  test -d "$candidate"
  test ! -e "$out"
  mkdir -p "$out"

  "$PY" "$P/export_v202_candidate_cache.py" \
    --baseline-cache "$RUN/audit/baseline/public_demo_holdout.npz" \
    --windows "$BASE/artifacts/adjust_bottle_windows_full" \
    --split-manifest "$RUN/public_demo_split.json" \
    --candidate "$candidate" --output "$out/public_demo_candidate.npz" \
    --chunks 4 --device cuda > "$out/public_demo_export.log" 2>&1

  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$out/public_demo_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/baseline/public_demo_holdout.npz" \
    --reuse-baseline-report "$RUN/audit/baseline/public_demo_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$out/public_demo_reward.json" --batch-size 32 --device cuda \
    > "$out/public_demo_reward.log" 2>&1
  touch "$out/COMPLETE"
done

touch "$RUN/CANDIDATE_AUDIT_COMPLETE"
echo V205_CANDIDATE_AUDIT_COMPLETE
