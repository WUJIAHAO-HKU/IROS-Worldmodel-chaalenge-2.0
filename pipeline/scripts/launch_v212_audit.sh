#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v212_v208_right_logit_long128_seed1411'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
LEFT="$JOINT/v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
RIGHT="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407/selected_right_expert"
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
PUBLIC_SPLIT="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
FAILURE_SPLIT="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407/audit/failure_holdout_split.json"
INSTRUCTIONS="$JOINT/reward_alignment/exact_instruction_map128.json"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

test -f "$RUN/TRAINING_COMPLETE"
test ! -e "$RUN/audit/baseline"
mkdir -p "$RUN/audit/baseline"

"$PY" "$P/export_strict_track2_p2_reward_cache.py" \
  --windows "$PUBLIC" --split-manifest "$PUBLIC_SPLIT" \
  --baseline-left "$LEFT" --baseline-right "$RIGHT" \
  --candidate-left "$LEFT" --candidate-right "$RIGHT" \
  --output "$RUN/audit/baseline/public_success_long128.npz" \
  --chunks 16 --max-sequences 32 --device cuda \
  >"$RUN/audit/baseline/public_success_export.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/baseline/public_success_long128.npz" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/baseline/public_success_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/baseline/public_success_reward.log" 2>&1

"$PY" "$P/export_strict_track2_p2_reward_cache.py" \
  --windows "$ONPOLICY" --split-manifest "$FAILURE_SPLIT" --instruction-map "$INSTRUCTIONS" \
  --baseline-left "$LEFT" --baseline-right "$RIGHT" \
  --candidate-left "$LEFT" --candidate-right "$RIGHT" \
  --output "$RUN/audit/baseline/public_failure_long128.npz" \
  --chunks 16 --max-sequences 32 --device cuda \
  >"$RUN/audit/baseline/public_failure_export.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/baseline/public_failure_long128.npz" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/baseline/public_failure_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/baseline/public_failure_reward.log" 2>&1
touch "$RUN/audit/baseline/COMPLETE"

for step in 50 100 150 200; do
  tag=$(printf 'step%04d' "$step")
  candidate=$(printf '%s/checkpoints/checkpoint_step_%06d' "$RUN" "$step")
  out="$RUN/audit/$tag"
  test -d "$candidate"
  mkdir -p "$out"
  "$PY" "$P/export_v202_candidate_cache.py" \
    --baseline-cache "$RUN/audit/baseline/public_success_long128.npz" \
    --windows "$PUBLIC" --split-manifest "$PUBLIC_SPLIT" \
    --candidate "$candidate" --output "$out/public_success_candidate.npz" \
    --chunks 16 --device cuda >"$out/public_success_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$out/public_success_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/baseline/public_success_long128.npz" \
    --reuse-baseline-report "$RUN/audit/baseline/public_success_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$out/public_success_reward.json" --batch-size 32 --device cuda \
    >"$out/public_success_reward.log" 2>&1

  "$PY" "$P/export_v202_candidate_cache.py" \
    --baseline-cache "$RUN/audit/baseline/public_failure_long128.npz" \
    --windows "$ONPOLICY" --split-manifest "$FAILURE_SPLIT" \
    --candidate "$candidate" --output "$out/public_failure_candidate.npz" \
    --chunks 16 --device cuda >"$out/public_failure_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$out/public_failure_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/baseline/public_failure_long128.npz" \
    --reuse-baseline-report "$RUN/audit/baseline/public_failure_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$out/public_failure_reward.json" --batch-size 32 --device cuda \
    >"$out/public_failure_reward.log" 2>&1
  touch "$out/COMPLETE"
done

"$PY" "$P/select_v212_long128_logit.py" \
  --run "$RUN" --registry "$REG" --output "$RUN/audit/v212_long128_logit_report.json" \
  >"$RUN/audit/v212_long128_logit_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V212_AUDIT_COMPLETE
