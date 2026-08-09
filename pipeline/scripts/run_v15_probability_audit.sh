#!/usr/bin/env bash
# Frozen-policy audit for rollout vs actor-recomputed diffusion log probabilities.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
rlinf_root=${RLINF_ROOT:-$root_dir/third_party/WorldArena-2.0/RL_env_benchmark}
actor_worker=$rlinf_root/rlinf/workers/actor/fsdp_actor_worker.py
mode=${RLINF_PPO_MODEL_MODE:-train}
source=${RLINF_BEHAVIOR_LOGPROB_SOURCE:-rollout}

if ! grep -q "probability_consistency_metrics" "$actor_worker"; then
  echo "RLinf probability patch is not applied; run pipeline/scripts/apply_rlinf_probability_patch.sh" >&2
  exit 3
fi

export RLINF_TRAIN_STEPS=${RLINF_TRAIN_STEPS:-1}
export RLINF_HORIZON_FRAMES=${RLINF_HORIZON_FRAMES:-16}
export RLINF_GROUP_SIZE=${RLINF_GROUP_SIZE:-4}
export RLINF_ACTOR_LR=0
export RLINF_SAVE_INTERVAL=${RLINF_SAVE_INTERVAL:-100}
export RLINF_PPO_MODEL_MODE=$mode
export RLINF_BEHAVIOR_LOGPROB_SOURCE=$source
export RLINF_PROBABILITY_AUDIT=true
export RLINF_LOG_PATH=${RLINF_LOG_PATH:-$root_dir/artifacts/rl_runs/v15_probability_audit_${mode}_${source}}

echo "probability audit: mode=$mode source=$source log=$RLINF_LOG_PATH"
bash "$root_dir/pipeline/scripts/run_v15_grpo_long_horizon.sh" "$@"
