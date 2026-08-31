#!/usr/bin/env bash
# Public-data-only Track 2 development run.  This script never reads final-eval seeds.
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

# Ray forks environment workers after tokenizer libraries may already be
# imported.  Defaulting this library-level switch to false removes the known
# fork warning/deadlock hazard without changing the policy or algorithm; an
# explicit caller setting remains authoritative.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

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
SAVE_INTERVAL="${TRACK2_SAVE_INTERVAL:-$MAX_STEPS}"
KEEP_LAST_CHECKPOINTS="${TRACK2_KEEP_LAST_CHECKPOINTS:-0}"
GROUP_SIZE="${TRACK2_GROUP_SIZE:-4}"
TOTAL_ENVS="${TRACK2_TOTAL_ENVS:-8}"
ACTOR_SEED="${TRACK2_ACTOR_SEED:-1249}"
ENV_SEED="${TRACK2_ENV_SEED:-0}"
ACTOR_LR="${TRACK2_ACTOR_LR:-5e-6}"
KL_BETA="${TRACK2_KL_BETA:-0.02}"
KL_PENALTY="${TRACK2_KL_PENALTY:-low_var_kl}"
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
RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT="${TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT:-true}"
TERMINAL_MODEL_ONLY_CHECKPOINT="${TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT:-true}"
REFERENCE_STATE_STORAGE="${TRACK2_REFERENCE_STATE_STORAGE:-disk}"
CKPT_PATH="${TRACK2_CKPT_PATH:-}"
RESUME_DIR="${TRACK2_RESUME_DIR:-}"
ALLOW_EXISTING_RUN="${TRACK2_ALLOW_EXISTING_RUN:-false}"
RAY_SYSTEM_CONFIG_JSON="${TRACK2_RAY_SYSTEM_CONFIG_JSON:-}"
START_STEP="${TRACK2_START_STEP:-0}"
REFERENCE_STATE_PATH_OVERRIDE="${TRACK2_REFERENCE_STATE_PATH_OVERRIDE:-}"
OPTIMIZER_RECOVERY_PATH="${TRACK2_OPTIMIZER_RECOVERY_PATH:-}"
MODEL_ONLY_RECOVERY="${TRACK2_MODEL_ONLY_RECOVERY:-0}"
CPUSET="${TRACK2_CPUSET:-0-2}"
ENABLE_SFT_CO_TRAIN="${TRACK2_ENABLE_SFT_CO_TRAIN:-false}"
SFT_DATA_PATH="${TRACK2_SFT_DATA_PATH:-}"
SFT_CONFIG_NAME="${TRACK2_SFT_CONFIG_NAME:-pi05_aloha_robotwin_head_adjust_bottle}"
SFT_LOSS_WEIGHT="${TRACK2_SFT_LOSS_WEIGHT:-0.1}"
SFT_BATCH_SIZE="${TRACK2_SFT_BATCH_SIZE:-1}"
SFT_NUM_WORKERS="${TRACK2_SFT_NUM_WORKERS:-0}"
SFT_SINGLE_ACTIVE_ARM="${TRACK2_SFT_SINGLE_ACTIVE_ARM:-0}"
USE_STRUCTURED_SFT_ACTION_LOSS="${TRACK2_USE_STRUCTURED_SFT_ACTION_LOSS:-false}"
USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS="${TRACK2_USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS:-false}"
SFT_PHYSICAL_ACTION_DIM="${TRACK2_SFT_PHYSICAL_ACTION_DIM:-}"
SFT_USE_ACTION_CHUNK_LOSS="${TRACK2_SFT_USE_ACTION_CHUNK_LOSS:-false}"
SFT_ACTIVE_JOINT_WEIGHT="${TRACK2_SFT_ACTIVE_JOINT_WEIGHT:-1.0}"
SFT_ACTIVE_GRIPPER_WEIGHT="${TRACK2_SFT_ACTIVE_GRIPPER_WEIGHT:-2.0}"
SFT_INACTIVE_KEEP_WEIGHT="${TRACK2_SFT_INACTIVE_KEEP_WEIGHT:-0.5}"
SFT_USE_MOTION_SAMPLE_WEIGHTING="${TRACK2_SFT_USE_MOTION_SAMPLE_WEIGHTING:-false}"
SFT_MOTION_WEIGHT_HORIZON="${TRACK2_SFT_MOTION_WEIGHT_HORIZON:-8}"
SFT_MOTION_WEIGHT_REFERENCE="${TRACK2_SFT_MOTION_WEIGHT_REFERENCE:-0.05}"
SFT_MOTION_MINIMUM_WEIGHT="${TRACK2_SFT_MOTION_MINIMUM_WEIGHT:-0.05}"
SFT_MOTION_WEIGHT_POWER="${TRACK2_SFT_MOTION_WEIGHT_POWER:-1.0}"
SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH="${TRACK2_SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH:-0}"
SFT_EXTRA_LOSS_WEIGHT="${TRACK2_SFT_EXTRA_LOSS_WEIGHT:-$SFT_LOSS_WEIGHT}"

weight_sync_args=()
checkpoint_args=()
resume_args=()
recovery_args=()
sft_args=()
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
if [[ "$KL_PENALTY" != "low_var_kl" && "$KL_PENALTY" != "k3" ]]; then
  echo "TRACK2_KL_PENALTY must be low_var_kl or k3 for conservative KL runs" >&2
  exit 2
fi
if [[ "$RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT" != "true" && "$RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT" != "false" ]]; then
  echo "TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT must be true or false" >&2
  exit 2
fi
if [[ "$TERMINAL_MODEL_ONLY_CHECKPOINT" != "true" && "$TERMINAL_MODEL_ONLY_CHECKPOINT" != "false" ]]; then
  echo "TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT must be true or false" >&2
  exit 2
fi
if [[ "$REFERENCE_STATE_STORAGE" != "memory" && "$REFERENCE_STATE_STORAGE" != "disk" ]]; then
  echo "TRACK2_REFERENCE_STATE_STORAGE must be memory or disk" >&2
  exit 2
fi
if [[ -n "$CKPT_PATH" ]]; then
  test -s "$CKPT_PATH"
  checkpoint_args+=("runner.ckpt_path=$CKPT_PATH")
fi
if (( START_STEP < 0 || START_STEP >= MAX_STEPS )); then
  echo "TRACK2_START_STEP must be in [0, TRACK2_MAX_STEPS)" >&2
  exit 2
fi
if (( START_STEP > 0 )); then
  test -s "$CKPT_PATH"
  test -s "$REFERENCE_STATE_PATH_OVERRIDE"
  recovery_args+=(
    "+actor.reference_state_path_override=$REFERENCE_STATE_PATH_OVERRIDE"
  )
  if [[ -n "$OPTIMIZER_RECOVERY_PATH" ]]; then
    test -s "$OPTIMIZER_RECOVERY_PATH"
    recovery_args+=(
      "+actor.optimizer_recovery_path=$OPTIMIZER_RECOVERY_PATH"
    )
  elif [[ "$MODEL_ONLY_RECOVERY" != 1 ]]; then
    echo "step recovery requires optimizer state unless TRACK2_MODEL_ONLY_RECOVERY=1" >&2
    exit 2
  fi
elif [[ -n "$REFERENCE_STATE_PATH_OVERRIDE" ]]; then
  test -s "$REFERENCE_STATE_PATH_OVERRIDE"
  recovery_args+=(
    "+actor.reference_state_path_override=$REFERENCE_STATE_PATH_OVERRIDE"
  )
fi
if [[ "$MODEL_ONLY_RECOVERY" != 0 && "$MODEL_ONLY_RECOVERY" != 1 ]]; then
  echo "TRACK2_MODEL_ONLY_RECOVERY must be 0 or 1" >&2
  exit 2
fi
if [[ "$ALLOW_EXISTING_RUN" != "true" && "$ALLOW_EXISTING_RUN" != "false" ]]; then
  echo "TRACK2_ALLOW_EXISTING_RUN must be true or false" >&2
  exit 2
fi
if [[ -n "$RESUME_DIR" ]]; then
  test -d "$RESUME_DIR/actor/dcp_checkpoint"
  resume_args+=("runner.resume_dir=$RESUME_DIR")
fi
if [[ -n "$CKPT_PATH" && -n "$RESUME_DIR" ]]; then
  echo "TRACK2_CKPT_PATH and TRACK2_RESUME_DIR are mutually exclusive" >&2
  exit 2
fi
if (( SAVE_INTERVAL < 1 || KEEP_LAST_CHECKPOINTS < 0 )); then
  echo "save interval must be positive and keep-last must be non-negative" >&2
  exit 2
fi
if ! taskset -c "$CPUSET" true >/dev/null 2>&1; then
  echo "TRACK2_CPUSET is invalid or unavailable: $CPUSET" >&2
  exit 2
fi
if [[ "$ENABLE_SFT_CO_TRAIN" != "true" && "$ENABLE_SFT_CO_TRAIN" != "false" ]]; then
  echo "TRACK2_ENABLE_SFT_CO_TRAIN must be true or false" >&2
  exit 2
fi
if [[ "$ENABLE_SFT_CO_TRAIN" == "true" ]]; then
  test -d "$SFT_DATA_PATH"
  if (( SFT_BATCH_SIZE < 1 )); then
    echo "TRACK2_SFT_BATCH_SIZE must be positive" >&2
    exit 2
  fi
  if (( SFT_NUM_WORKERS < 0 )); then
    echo "TRACK2_SFT_NUM_WORKERS must be non-negative" >&2
    exit 2
  fi
  if (( SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH < 0 )); then
    echo "TRACK2_SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH must be non-negative" >&2
    exit 2
  fi
  if [[ "$SFT_SINGLE_ACTIVE_ARM" != "0" && "$SFT_SINGLE_ACTIVE_ARM" != "1" ]]; then
    echo "TRACK2_SFT_SINGLE_ACTIVE_ARM must be 0 or 1" >&2
    exit 2
  fi
  if [[ "$USE_STRUCTURED_SFT_ACTION_LOSS" != "true" && "$USE_STRUCTURED_SFT_ACTION_LOSS" != "false" ]]; then
    echo "TRACK2_USE_STRUCTURED_SFT_ACTION_LOSS must be true or false" >&2
    exit 2
  fi
  if [[ "$USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS" != "true" && "$USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS" != "false" ]]; then
    echo "TRACK2_USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS must be true or false" >&2
    exit 2
  fi
  if [[ "$USE_STRUCTURED_SFT_ACTION_LOSS" == "true" && "$USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS" == "true" ]]; then
    echo "single-arm and mixed-arm structured SFT losses are mutually exclusive" >&2
    exit 2
  fi
  if [[ -n "$SFT_PHYSICAL_ACTION_DIM" ]]; then
    if [[ ! "$SFT_PHYSICAL_ACTION_DIM" =~ ^[1-9][0-9]*$ ]]; then
      echo "TRACK2_SFT_PHYSICAL_ACTION_DIM must be a positive integer" >&2
      exit 2
    fi
    if [[ "$USE_STRUCTURED_SFT_ACTION_LOSS" == "true" ]]; then
      echo "TRACK2_SFT_PHYSICAL_ACTION_DIM cannot be combined with structured SFT loss" >&2
      exit 2
    fi
    if [[ "$USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS" == "true" ]]; then
      echo "TRACK2_SFT_PHYSICAL_ACTION_DIM cannot be combined with mixed-arm structured SFT loss" >&2
      exit 2
    fi
  fi
  if [[ "$SFT_USE_ACTION_CHUNK_LOSS" != "true" && "$SFT_USE_ACTION_CHUNK_LOSS" != "false" ]]; then
    echo "TRACK2_SFT_USE_ACTION_CHUNK_LOSS must be true or false" >&2
    exit 2
  fi
  sft_args+=(
    "+actor.enable_sft_co_train=true"
    "+actor.sft_data_path=$SFT_DATA_PATH"
    "+actor.config_name=$SFT_CONFIG_NAME"
    "+actor.sft_loss_weight=$SFT_LOSS_WEIGHT"
    "+actor.sft_batch_size=$SFT_BATCH_SIZE"
    "+actor.sft_num_workers=$SFT_NUM_WORKERS"
    "+actor.sft_single_active_arm=$SFT_SINGLE_ACTIVE_ARM"
    "+actor.use_structured_sft_action_loss=$USE_STRUCTURED_SFT_ACTION_LOSS"
    "+actor.use_mixed_arm_structured_sft_action_loss=$USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS"
    "+actor.sft_use_action_chunk_loss=$SFT_USE_ACTION_CHUNK_LOSS"
    "+actor.sft_active_joint_weight=$SFT_ACTIVE_JOINT_WEIGHT"
    "+actor.sft_active_gripper_weight=$SFT_ACTIVE_GRIPPER_WEIGHT"
    "+actor.sft_inactive_keep_weight=$SFT_INACTIVE_KEEP_WEIGHT"
    "+actor.sft_gripper_activity_weight=${TRACK2_SFT_GRIPPER_ACTIVITY_WEIGHT:-0.25}"
    "+actor.sft_use_motion_sample_weighting=$SFT_USE_MOTION_SAMPLE_WEIGHTING"
    "+actor.sft_motion_weight_horizon=$SFT_MOTION_WEIGHT_HORIZON"
    "+actor.sft_motion_weight_reference=$SFT_MOTION_WEIGHT_REFERENCE"
    "+actor.sft_motion_minimum_weight=$SFT_MOTION_MINIMUM_WEIGHT"
    "+actor.sft_motion_weight_power=$SFT_MOTION_WEIGHT_POWER"
    "+actor.sft_extra_updates_per_global_batch=$SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH"
    "+actor.sft_extra_loss_weight=$SFT_EXTRA_LOSS_WEIGHT"
  )
  if [[ -n "$SFT_PHYSICAL_ACTION_DIM" ]]; then
    sft_args+=("+actor.sft_physical_action_dim=$SFT_PHYSICAL_ACTION_DIM")
  fi
fi

if (( EPISODE_STEPS < 8 || EPISODE_STEPS % 8 != 0 )); then
  echo "TRACK2_MAX_EPISODE_STEPS must be a positive multiple of the official 8-action chunk" >&2
  exit 2
fi
if (( ROLLOUT_STEPS < 8 || ROLLOUT_STEPS % 8 != 0 )); then
  echo "TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH must be a positive multiple of the official 8-action chunk" >&2
  exit 2
fi

if [[ "$ALLOW_EXISTING_RUN" == "true" ]]; then
  test -d "$RUN/audit"
else
  test ! -e "$RUN"
  mkdir -p "$RUN/audit"
fi

if [[ "$ALLOW_EXISTING_RUN" != "true" ]]; then
  "$GO1" -c 'import json,sys; split=json.load(open(sys.argv[1])); assert len(split["train_episodes"]) == 112; assert len(split["validation_episodes"]) == 16; print("public_split_ok")' \
    "$PUBLIC_WINDOWS/split_manifest.json" > "$RUN/audit/public_split_check.txt"
fi
curl -fsS "$BRIDGE_URL/health" | grep -q '"status":"ready"'
curl -fsS "$WAM_URL/v1/health" | grep -q "$MODEL_VERSION"
if [[ "$ALLOW_EXISTING_RUN" != "true" ]]; then
  sha256sum "$OPENPI_CKPT/model.safetensors" "$OPENPI_CKPT/metadata.pt" \
    "$OPENPI_CKPT/rlinf/robotwin_headcam_adjust_bottle/norm_stats.json" "$REWARD" \
    "$PARENT_MODEL" > "$RUN/audit/frozen_inputs_sha256.txt"
  if [[ -n "$CKPT_PATH" ]]; then
    sha256sum "$CKPT_PATH" >> "$RUN/audit/frozen_inputs_sha256.txt"
  fi
  if [[ -n "$REFERENCE_STATE_PATH_OVERRIDE" ]]; then
    sha256sum "$REFERENCE_STATE_PATH_OVERRIDE" \
      >> "$RUN/audit/frozen_inputs_sha256.txt"
  fi
  if [[ "$ENABLE_SFT_CO_TRAIN" == "true" ]]; then
    find "$SFT_DATA_PATH" -type f -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN/audit/sft_dataset_sha256.txt"
  fi
  sha256sum "$RLINF/rlinf/workers/actor/fsdp_actor_worker.py" \
    "$RLINF/rlinf/workers/rollout/hf/huggingface_worker.py" \
    "$RLINF/rlinf/runners/embodied_runner.py" \
    > "$RUN/audit/rlinf_source_sha256.txt"
fi

# The integrity audit above reads the 7.5 GB frozen policy into the cgroup page
# cache.  It is no longer needed at this point and retaining it can make the
# subsequent actor/rollout model construction exceed the 62 GB cgroup limit.
# This is only a best-effort cache hint: it never changes model bytes or the
# already-written SHA256 audit record.
if [[ "$ALLOW_EXISTING_RUN" != "true" ]]; then
cache_evict_paths=("$OPENPI_CKPT/model.safetensors")
if [[ -n "$REFERENCE_STATE_PATH_OVERRIDE" ]]; then
  cache_evict_paths+=("$REFERENCE_STATE_PATH_OVERRIDE")
fi
"$GO1" - "${cache_evict_paths[@]}" <<'PY'
import os
import sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PY
fi

"$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_before.log" 2>&1 || true
# A local Ray default reserves ~26% of the 66 GB cgroup (about 17 GB) for
# plasma.  FSDP needs that headroom while it materializes the frozen policy,
# whereas this one-step GRPO run does not move multi-GB objects through plasma.
ray_start_args=(--head --object-store-memory=6442450944 --disable-usage-stats)
if [[ -n "$RAY_SYSTEM_CONFIG_JSON" ]]; then
  ray_start_args+=("--system-config=$RAY_SYSTEM_CONFIG_JSON")
fi
taskset -c "$CPUSET" "$PY" -m ray.scripts.scripts start "${ray_start_args[@]}" \
  > "$RUN/ray_start.log" 2>&1
cd "$RLINF"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" OPENPI_CKPT_PATH="$OPENPI_CKPT" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 CATCH_SYSTEM_FAILURE="$CATCH_SYSTEM_FAILURE" \
taskset -c "$CPUSET" "$PY" "$RLINF/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.max_steps="$MAX_STEPS" runner.max_epochs=1000 runner.save_interval="$SAVE_INTERVAL" \
  +runner.keep_last_checkpoints="$KEEP_LAST_CHECKPOINTS" +runner.start_step="$START_STEP" \
  env.train.total_num_envs="$TOTAL_ENVS" env.train.group_size="$GROUP_SIZE" env.train.seed="$ENV_SEED" env.train.max_episode_steps="$EPISODE_STEPS" env.train.max_steps_per_rollout_epoch="$ROLLOUT_STEPS" env.train.enable_offload="$ENV_OFFLOAD" \
  env.train.http.server_url="$BRIDGE_URL" \
  env.train.VAE_path="$ADAPTER/Wan2.2_VAE.pth" env.train.model_path="$ADAPTER/dit_model.safetensors" env.train.initial_image_path="$ADAPTER/dataset/" \
  algorithm.group_size="$GROUP_SIZE" algorithm.rollout_epoch="$ROLLOUT_EPOCH" algorithm.eval_rollout_epoch=0 algorithm.kl_beta="$KL_BETA" algorithm.kl_penalty="$KL_PENALTY" algorithm.filter_rewards="$FILTER_REWARDS" \
  actor.seed="$ACTOR_SEED" actor.global_batch_size="$ACTOR_GLOBAL_BATCH_SIZE" actor.micro_batch_size=1 actor.optim.lr="$ACTOR_LR" \
  +actor.release_reference_before_terminal_checkpoint="$RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT" \
  +actor.terminal_model_only_checkpoint="$TERMINAL_MODEL_ONLY_CHECKPOINT" \
  +actor.reference_state_storage="$REFERENCE_STATE_STORAGE" \
  actor.enable_offload="$ACTOR_OFFLOAD" actor.fsdp_config.cpu_offload="$FSDP_CPU_OFFLOAD" rollout.enable_offload=true \
  "${checkpoint_args[@]}" \
  "${resume_args[@]}" \
  "${recovery_args[@]}" \
  "${sft_args[@]}" \
  "${weight_sync_args[@]}" \
  > "$RUN/launcher.log" 2>&1

if grep -aqE 'CUDA out of memory|OutOfMemoryError|HTTP.*(500|502|503|504)' "$RUN/launcher.log"; then
  exit 7
fi
test -s "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${MAX_STEPS}/actor/model_state_dict/full_weights.pt"
"$PY" -c 'import sys, zipfile; assert zipfile.is_zipfile(sys.argv[1]), "invalid or truncated PyTorch checkpoint"' \
  "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${MAX_STEPS}/actor/model_state_dict/full_weights.pt"
printf 'V173_PUBLIC_KL_FOUR_STEP_COMPLETE\n'
