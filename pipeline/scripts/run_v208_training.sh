#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
P="$BASE/pipeline/scripts"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/miniconda3/envs/go1/bin/python'
PARENT="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/selected_right_expert"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
MIXED="$JOINT/v163_mixed_reward_windows"
export PYTHONPATH="$BASE/pipeline:$BASE/pipeline/scripts:$RLINF"

test -f "$RUN/audit/baseline/COMPLETE"
test ! -e "$RUN/checkpoints"

"$PY" "$P/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$MIXED" --split-manifest "$RUN/training_split_arm_prompt_fixed.json" \
  --init-checkpoint "$PARENT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/checkpoints" \
  --steps 1200 --checkpoint-interval 200 --batch-size 1 \
  --chunks 4 --chunk-stride 8 --arm-filter right \
  --learning-rate 1e-7 \
  --reward-objective probability --reward-probability-scale-floor 0.01 \
  --reward-loss-weight 0.1 --reward-delta-weight 2.0 --reward-terminal-weight 10.0 \
  --right-weight 1.0 --success-weight 1.25 \
  --late-weight 4.0 --late-start 64 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --max-grad-norm 1.0 \
  --seed 1407 --device cuda >"$RUN/training.log" 2>&1

touch "$RUN/TRAINING_COMPLETE"
echo V208_TRAINING_COMPLETE
