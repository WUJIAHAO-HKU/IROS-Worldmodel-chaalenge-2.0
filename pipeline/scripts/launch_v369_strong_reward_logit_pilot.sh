#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v369_v354_strong_reward_logit_recursive_pilot_seed1533_20260822"
PY=/root/miniconda3/envs/go1/bin/python
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
RESET="$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
test -s "$RUN/release_registration.json"
test ! -e "$RUN/checkpoints"
taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json" \
  --init-checkpoint "$JOINT/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/best" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/checkpoints" --steps 50 --checkpoint-interval 25 --batch-size 1 \
  --chunks 8 --chunk-stride 8 --arm-filter right --learning-rate 5e-8 \
  --reward-objective logit --reward-probability-scale-floor 0.01 \
  --reward-loss-weight 0.01 --reward-delta-weight 0.5 --reward-terminal-weight 2.0 \
  --right-weight 1.0 --success-weight 1.0 --late-weight 12.0 --late-start 48 \
  --terminal-visual-weight 2.0 --prompts-per-arm 4 --max-grad-norm 1.0 \
  --seed 1533 --device cuda >"$RUN/training.log" 2>&1
touch "$RUN/TRAINING_COMPLETE"
printf 'V369_TRAINING_COMPLETE\n'
