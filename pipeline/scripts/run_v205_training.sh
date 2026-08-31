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

test -f "$RUN/audit/baseline/COMPLETE"
test ! -e "$RUN/checkpoints"

"$PY" "$P/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$BASE/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$RUN/public_demo_split.json" \
  --init-checkpoint "$PARENT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/checkpoints" \
  --steps 120 --checkpoint-interval 30 --batch-size 1 \
  --chunks 4 --chunk-stride 8 --arm-filter right \
  --learning-rate 1e-7 \
  --reward-objective probability --reward-probability-scale-floor 0.01 \
  --reward-loss-weight 0.05 --reward-delta-weight 2.0 --reward-terminal-weight 10.0 \
  --right-weight 1.0 --success-weight 1.0 \
  --late-weight 6.0 --late-start 64 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --max-grad-norm 1.0 \
  --seed 1405 --device cuda \
  > "$RUN/training.log" 2>&1

touch "$RUN/TRAINING_COMPLETE"
echo V205_TRAINING_COMPLETE
