#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405"
P="$BASE/pipeline/scripts"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/miniconda3/envs/go1/bin/python'
PARENT="$JOINT/v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$BASE/pipeline/scripts:$RLINF"

test -f "$RUN/public_demo_split.json"
test ! -e "$RUN/audit/baseline/public_demo_holdout.npz"

"$PY" "$P/export_strict_track2_p2_reward_cache.py" \
  --windows "$BASE/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$RUN/public_demo_split.json" \
  --baseline-left "$PARENT" --baseline-right "$PARENT" \
  --candidate-left "$PARENT" --candidate-right "$PARENT" \
  --output "$RUN/audit/baseline/public_demo_holdout.npz" \
  --chunks 4 --max-sequences 40 --device cuda \
  > "$RUN/audit/baseline/public_demo_export.log" 2>&1

"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/baseline/public_demo_holdout.npz" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/baseline/public_demo_reward.json" \
  --batch-size 32 --device cuda \
  > "$RUN/audit/baseline/public_demo_reward.log" 2>&1

touch "$RUN/audit/baseline/COMPLETE"
echo V205_BASELINE_AUDIT_COMPLETE
