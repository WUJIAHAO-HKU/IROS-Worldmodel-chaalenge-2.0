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
V208="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
MIXED="$JOINT/v163_mixed_reward_windows"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

restart_services() {
  bash "$P/restart_v209_services.sh" start >"$REG/restart_v209_after_v212.log" 2>&1 || true
}
trap restart_services EXIT

"$PY" "$P/prepare_v212_right_logit_long128.py"
for name in wm_v209_bridge wm_v209_gpu; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
  sleep 1
done
if ss -ltn | grep -qE ':(8004|18083) '; then
  echo 'v209 ports did not stop cleanly' >&2
  exit 3
fi

"$PY" "$P/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$MIXED" --split-manifest "$V208/training_split_arm_prompt_fixed.json" \
  --init-checkpoint "$V208/selected_right_expert" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/checkpoints" \
  --steps 200 --checkpoint-interval 50 --batch-size 1 \
  --chunks 16 --chunk-stride 8 --arm-filter right \
  --learning-rate 5e-8 \
  --reward-objective logit --reward-probability-scale-floor 0.01 \
  --reward-loss-weight 0.02 --reward-delta-weight 2.0 --reward-terminal-weight 4.0 \
  --right-weight 1.0 --success-weight 3.0 \
  --late-weight 4.0 --late-start 64 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --max-grad-norm 1.0 \
  --seed 1411 --device cuda >"$RUN/training.log" 2>&1

touch "$RUN/TRAINING_COMPLETE"
echo V212_TRAINING_COMPLETE
