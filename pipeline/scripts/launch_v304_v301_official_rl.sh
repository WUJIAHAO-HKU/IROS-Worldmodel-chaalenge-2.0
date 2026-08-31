#!/usr/bin/env bash
set -euo pipefail
export TRACK2_LAUNCH_NAME='v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821'
export TRACK2_LAUNCH_RELEASE='/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v303_v301_batched_gates_seed1482_20260821'
export TRACK2_LAUNCH_MODEL_VERSION='track2-v301-batched-terminal-frame-mirror-v295'
export TRACK2_LAUNCH_PREPARE_SCRIPT='prepare_v304_v301_official_rl.py'
export TRACK2_LAUNCH_SERVICE_SCRIPT='restart_v301_services.sh'
export TRACK2_LAUNCH_ACCEPT_MARKER='V304_TRAINING_ACCEPTED'
exec bash '/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v300_v295_official_rl.sh'
