#!/usr/bin/env bash
# Conservative formal GRPO recipe for the packaged Track 2 v15 world model.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
steps=${RLINF_TRAIN_STEPS:-20}
save_interval=${RLINF_SAVE_INTERVAL:-$steps}
actor_lr=${RLINF_ACTOR_LR:-5.0e-7}
clip_ratio=${RLINF_CLIP_RATIO:-0.1}
clip_grad=${RLINF_CLIP_GRAD:-0.5}
log_path=${RLINF_LOG_PATH:-$root_dir/artifacts/rl_runs/v15_grpo_conservative${steps}}
resume_dir=${RLINF_RESUME_DIR:-null}

# OpenPI imports JAX for checkpoint/config utilities. Keep JAX on CPU so the
# v15 service, policy actor, rollout worker, and reward model can share the GPU.
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export USE_TF=${USE_TF:-0}

exec bash "$root_dir/pipeline/scripts/run_public_rlinf_track2.sh" \
  "runner.max_steps=$steps" \
  "runner.max_epochs=$steps" \
  "runner.save_interval=$save_interval" \
  "runner.resume_dir=$resume_dir" \
  "runner.logger.log_path=$log_path" \
  "actor.optim.lr=$actor_lr" \
  "actor.optim.clip_grad=$clip_grad" \
  "algorithm.clip_ratio_low=$clip_ratio" \
  "algorithm.clip_ratio_high=$clip_ratio" \
  "$@"
