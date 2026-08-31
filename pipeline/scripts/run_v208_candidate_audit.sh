#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
P="$BASE/pipeline/scripts"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/miniconda3/envs/go1/bin/python'
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
PUBLIC_SPLIT="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
MIXED="$JOINT/v163_mixed_reward_windows"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
FAILURE_SPLIT="$RUN/audit/failure_holdout_split.json"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$BASE/pipeline/scripts:$RLINF"

test -f "$RUN/TRAINING_COMPLETE"
test -f "$RUN/audit/baseline/COMPLETE"
for step in 200 400 600 800 1000 1200; do
  tag=$(printf 'step%04d' "$step")
  candidate=$(printf '%s/checkpoints/checkpoint_step_%06d' "$RUN" "$step")
  out="$RUN/audit/$tag"
  test -d "$candidate"
  test ! -e "$out"
  mkdir -p "$out"

  "$PY" "$P/export_v202_candidate_cache.py" \
    --baseline-cache "$RUN/audit/baseline/public_success_holdout.npz" \
    --windows "$PUBLIC" --split-manifest "$PUBLIC_SPLIT" \
    --candidate "$candidate" --output "$out/public_success_candidate.npz" \
    --chunks 4 --device cuda >"$out/public_success_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$out/public_success_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/baseline/public_success_holdout.npz" \
    --reuse-baseline-report "$RUN/audit/baseline/public_success_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$out/public_success_reward.json" --batch-size 32 --device cuda \
    >"$out/public_success_reward.log" 2>&1

  "$PY" "$P/export_v202_candidate_cache.py" \
    --baseline-cache "$RUN/audit/baseline/public_failure_holdout.npz" \
    --windows "$ONPOLICY" --split-manifest "$FAILURE_SPLIT" \
    --candidate "$candidate" --output "$out/public_failure_candidate.npz" \
    --chunks 4 --device cuda >"$out/public_failure_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$out/public_failure_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/baseline/public_failure_holdout.npz" \
    --reuse-baseline-report "$RUN/audit/baseline/public_failure_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$out/public_failure_reward.json" --batch-size 32 --device cuda \
    >"$out/public_failure_reward.log" 2>&1
  touch "$out/COMPLETE"
done
touch "$RUN/CANDIDATE_AUDIT_COMPLETE"
echo V208_CANDIDATE_AUDIT_COMPLETE
