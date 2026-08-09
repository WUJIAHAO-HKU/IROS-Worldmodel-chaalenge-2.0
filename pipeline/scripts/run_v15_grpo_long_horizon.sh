#!/usr/bin/env bash
# Multi-chunk Track 2 GRPO experiment with official-reward progress shaping.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
steps=${RLINF_TRAIN_STEPS:-100}
horizon=${RLINF_HORIZON_FRAMES:-64}
group_size=${RLINF_GROUP_SIZE:-4}
save_interval=${RLINF_SAVE_INTERVAL:-$steps}
actor_lr=${RLINF_ACTOR_LR:-2.0e-7}
resume_dir=${RLINF_RESUME_DIR:-null}
log_path=${RLINF_LOG_PATH:-$root_dir/artifacts/rl_runs/v15_grpo_long_horizon_g${group_size}_s${steps}}

if (( horizon < 16 || horizon % 8 != 0 )); then
  echo "RLINF_HORIZON_FRAMES must be a multiple of 8 and at least 16" >&2
  exit 2
fi
if (( group_size < 2 )); then
  echo "RLINF_GROUP_SIZE must be at least 2" >&2
  exit 2
fi

export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export USE_TF=${USE_TF:-0}

exec bash "$root_dir/pipeline/scripts/run_public_rlinf_track2.sh" \
  "runner.max_steps=$steps" \
  "runner.max_epochs=$steps" \
  "runner.save_interval=$save_interval" \
  "runner.resume_dir=$resume_dir" \
  "runner.logger.log_path=$log_path" \
  "env.train.total_num_envs=$group_size" \
  "env.train.group_size=$group_size" \
  "env.train.max_steps_per_rollout_epoch=$horizon" \
  "env.train.max_episode_steps=$horizon" \
  "+env.train.initialize_reward_baseline=true" \
  "+env.train.delta_reward_weight=1.0" \
  "+env.train.terminal_progress_weight=1.0" \
  "+env.train.peak_progress_weight=0.5" \
  "+env.train.positive_delta_reward_weight=0.25" \
  "+env.train.reward_clip=1.0" \
  "algorithm.group_size=$group_size" \
  "actor.global_batch_size=$group_size" \
  "actor.optim.lr=$actor_lr" \
  "actor.optim.clip_grad=0.5" \
  "algorithm.clip_ratio_low=0.1" \
  "algorithm.clip_ratio_high=0.1" \
  "$@"
