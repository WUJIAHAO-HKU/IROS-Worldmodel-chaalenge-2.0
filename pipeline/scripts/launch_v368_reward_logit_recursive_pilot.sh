#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v368_v354_right_reward_logit_recursive_pilot_seed1532_20260822"
PY=/root/miniconda3/envs/go1/bin/python
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PARENT="$JOINT/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/best"
SPLIT="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
RESET="$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"

test -s "$RUN/release_registration.json"
test ! -e "$RUN/checkpoints"
if pgrep -af '[t]rain_embodied_agent.py|[e]val_embodied_agent.py|[t]rain_multichunk_reward_aligned_autoregressive_unet.py' >/dev/null; then
  printf 'Refusing to contend with an active training/evaluation process\n' >&2
  exit 3
fi

export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$SPLIT" \
  --init-checkpoint "$PARENT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/checkpoints" \
  --steps 20 --checkpoint-interval 10 --batch-size 1 \
  --chunks 8 --chunk-stride 8 --arm-filter right \
  --learning-rate 5e-8 \
  --reward-objective logit --reward-probability-scale-floor 0.01 \
  --reward-loss-weight 0.002 --reward-delta-weight 0.5 --reward-terminal-weight 2.0 \
  --right-weight 1.0 --success-weight 1.0 \
  --late-weight 12.0 --late-start 48 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --max-grad-norm 1.0 \
  --seed 1532 --device cuda >"$RUN/training.log" 2>&1

test -s "$RUN/checkpoints/checkpoint_step_000010/model.pt"
test -s "$RUN/checkpoints/checkpoint_step_000020/model.pt"
touch "$RUN/TRAINING_COMPLETE"
printf 'V368_TRAINING_COMPLETE\n'
