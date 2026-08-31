#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v204_v202_effectivekl_h200_r2_step8_lr1e5_beta005_seed1404_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PARENT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v202_v201_public_terminal_reward_calibration_seed1402/selected_model/model.pt"

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION='track2-v202-public-terminal-calibrated-step75'
export TRACK2_BRIDGE_URL='http://127.0.0.1:18083' TRACK2_WAM_URL='http://127.0.0.1:8004'
export TRACK2_PARENT_MODEL="$PARENT"
export TRACK2_MAX_STEPS=8 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1404 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=1e-5 TRACK2_KL_BETA=0.05 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200
export TRACK2_ROLLOUT_EPOCH=2 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=400
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TOKENIZERS_PARALLELISM=false

exec bash "$ROOT/pipeline/scripts/run_strict_track2_conservative_kl.sh" > "$REG/launcher.screen.log" 2>&1
