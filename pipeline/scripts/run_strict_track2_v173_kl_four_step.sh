#!/usr/bin/env bash
# Public-data-only Track 2 development run.  This script never reads final-eval seeds.
set -euo pipefail

# Ray forks environment workers after tokenizer libraries may already be
# imported.  Defaulting this library-level switch to false removes the known
# fork warning/deadlock hazard without changing the policy or algorithm; an
# explicit caller setting remains authoritative.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_RL_PYTHON:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$AUDIT/official_deps/diffsynth_2a2e05f"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
ADAPTER="$AUDIT/v15_frozen_input_adapter"
PUBLIC_WINDOWS="$JOINT/onpolicy_windows_full128_stride4"
RUN="${TRACK2_RUN:-$AUDIT/runs/v173_public_kl_four_step_seed1248_20260815}"
MODEL_VERSION="${TRACK2_MODEL_VERSION:-track2-v173-public-terminal-focus}"
BRIDGE_URL="${TRACK2_BRIDGE_URL:-http://127.0.0.1:18081}"
WAM_URL="${TRACK2_WAM_URL:-http://127.0.0.1:8002}"
PARENT_MODEL="${TRACK2_PARENT_MODEL:-$JOINT/v173_v172_public_long16_terminal_focus/best/model.pt}"
MAX_STEPS="${TRACK2_MAX_STEPS:-4}"
GROUP_SIZE="${TRACK2_GROUP_SIZE:-4}"
TOTAL_ENVS="${TRACK2_TOTAL_ENVS:-8}"
ACTOR_SEED="${TRACK2_ACTOR_SEED:-1249}"
ENV_SEED="${TRACK2_ENV_SEED:-0}"
ACTOR_LR="${TRACK2_ACTOR_LR:-5e-6}"
KL_BETA="${TRACK2_KL_BETA:-0.02}"
EPISODE_STEPS="${TRACK2_MAX_EPISODE_STEPS:-8}"
ROLLOUT_STEPS="${TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH:-$EPISODE_STEPS}"
ROLLOUT_EPOCH="${TRACK2_ROLLOUT_EPOCH:-1}"
FILTER_REWARDS="${TRACK2_FILTER_REWARDS:-false}"
ACTOR_GLOBAL_BATCH_SIZE="${TRACK2_ACTOR_GLOBAL_BATCH_SIZE:-8}"
# The only prior run that completed an actor update used the resident FSDP actor.
# Keep that proven topology for the v173 pilot; offloading remains opt-in.
ACTOR_OFFLOAD="${TRACK2_ACTOR_OFFLOAD:-false}"
ENV_OFFLOAD="${TRACK2_ENV_OFFLOAD:-false}"
# Resource-only FSDP setting.  It keeps the frozen policy and algorithm intact,
# but avoids holding all FSDP parameters on the single 16 GB GPU during the
# actor/rollout overlap.
FSDP_CPU_OFFLOAD="${TRACK2_FSDP_CPU_OFFLOAD:-false}"
WEIGHT_SYNC_TRANSPORT_DEVICE="${TRACK2_WEIGHT_SYNC_TRANSPORT_DEVICE:-}"
CATCH_SYSTEM_FAILURE="${TRACK2_CATCH_SYSTEM_FAILURE:-0}"

weight_sync_args=()
if [[ -n "$WEIGHT_SYNC_TRANSPORT_DEVICE" ]]; then
  if [[ "$WEIGHT_SYNC_TRANSPORT_DEVICE" != "cpu" && "$WEIGHT_SYNC_TRANSPORT_DEVICE" != "cuda" ]]; then
    echo "TRACK2_WEIGHT_SYNC_TRANSPORT_DEVICE must be cpu or cuda" >&2
    exit 2
  fi
  weight_sync_args+=("+weight_syncer.patch.transport_device=$WEIGHT_SYNC_TRANSPORT_DEVICE")
fi
if [[ "$CATCH_SYSTEM_FAILURE" != "0" && "$CATCH_SYSTEM_FAILURE" != "1" ]]; then
  echo "TRACK2_CATCH_SYSTEM_FAILURE must be 0 or 1" >&2
  exit 2
fi
if [[ "$FSDP_CPU_OFFLOAD" != "true" && "$FSDP_CPU_OFFLOAD" != "false" ]]; then
  echo "TRACK2_FSDP_CPU_OFFLOAD must be true or false" >&2
  exit 2
fi

if (( EPISODE_STEPS < 8 || EPISODE_STEPS % 8 != 0 )); then
  echo "TRACK2_MAX_EPISODE_STEPS must be a positive multiple of the official 8-action chunk" >&2
  exit 2
fi
if (( ROLLOUT_STEPS < 8 || ROLLOUT_STEPS % 8 != 0 )); then
  echo "TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH must be a positive multiple of the official 8-action chunk" >&2
  exit 2
fi

test ! -e "$RUN"
mkdir -p "$RUN/audit"

"$GO1" -c 'import json,sys; split=json.load(open(sys.argv[1])); assert len(split["train_episodes"]) == 112; assert len(split["validation_episodes"]) == 16; print("public_split_ok")' \
  "$PUBLIC_WINDOWS/split_manifest.json" > "$RUN/audit/public_split_check.txt"
curl -fsS "$BRIDGE_URL/health" | grep -q '"status":"ready"'
curl -fsS "$WAM_URL/v1/health" | grep -q "$MODEL_VERSION"
sha256sum "$OPENPI_CKPT/model.safetensors" "$OPENPI_CKPT/metadata.pt" \
  "$OPENPI_CKPT/rlinf/robotwin_headcam_adjust_bottle/norm_stats.json" "$REWARD" \
  "$PARENT_MODEL" > "$RUN/audit/frozen_inputs_sha256.txt"
sha256sum "$RLINF/rlinf/workers/actor/fsdp_actor_worker.py" \
  "$RLINF/rlinf/workers/rollout/hf/huggingface_worker.py" \
  > "$RUN/audit/rlinf_source_sha256.txt"

# The integrity audit above reads the 7.5 GB frozen policy into the cgroup page
# cache.  It is no longer needed at this point and retaining it can make the
# subsequent actor/rollout model construction exceed the 62 GB cgroup limit.
# This is only a best-effort cache hint: it never changes model bytes or the
# already-written SHA256 audit record.
"$GO1" - "$OPENPI_CKPT/model.safetensors" <<'PY'
import os
import sys

fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
finally:
    os.close(fd)
PY

"$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_before.log" 2>&1 || true
# A local Ray default reserves ~26% of the 66 GB cgroup (about 17 GB) for
# plasma.  FSDP needs that headroom while it materializes the frozen policy,
# whereas this one-step GRPO run does not move multi-GB objects through plasma.
"$PY" -m ray.scripts.scripts start --head --object-store-memory=6442450944 --disable-usage-stats \
  > "$RUN/ray_start.log" 2>&1
cd "$RLINF"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" OPENPI_CKPT_PATH="$OPENPI_CKPT" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 CATCH_SYSTEM_FAILURE="$CATCH_SYSTEM_FAILURE" \
"$PY" "$RLINF/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.max_steps="$MAX_STEPS" runner.max_epochs=1000 runner.save_interval="$MAX_STEPS" \
  env.train.total_num_envs="$TOTAL_ENVS" env.train.group_size="$GROUP_SIZE" env.train.seed="$ENV_SEED" env.train.max_episode_steps="$EPISODE_STEPS" env.train.max_steps_per_rollout_epoch="$ROLLOUT_STEPS" env.train.enable_offload="$ENV_OFFLOAD" \
  env.train.http.server_url="$BRIDGE_URL" \
  env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" env.train.model_path="$ADAPTER/dit_model.safetensors" env.train.initial_image_path="$ADAPTER/dataset/" \
  algorithm.group_size="$GROUP_SIZE" algorithm.rollout_epoch="$ROLLOUT_EPOCH" algorithm.eval_rollout_epoch=0 algorithm.kl_beta="$KL_BETA" algorithm.filter_rewards="$FILTER_REWARDS" \
  actor.seed="$ACTOR_SEED" actor.global_batch_size="$ACTOR_GLOBAL_BATCH_SIZE" actor.micro_batch_size=1 actor.optim.lr="$ACTOR_LR" \
  actor.enable_offload="$ACTOR_OFFLOAD" actor.fsdp_config.cpu_offload="$FSDP_CPU_OFFLOAD" rollout.enable_offload=true \
  "${weight_sync_args[@]}" \
  > "$RUN/launcher.log" 2>&1

if grep -aqE 'CUDA out of memory|OutOfMemoryError|HTTP.*(500|502|503|504)' "$RUN/launcher.log"; then
  exit 7
fi
test -s "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${MAX_STEPS}/actor/model_state_dict/full_weights.pt"
"$PY" -c 'import sys, zipfile; assert zipfile.is_zipfile(sys.argv[1]), "invalid or truncated PyTorch checkpoint"' \
  "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${MAX_STEPS}/actor/model_state_dict/full_weights.pt"
printf 'V173_PUBLIC_KL_FOUR_STEP_COMPLETE\n'
