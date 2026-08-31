#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
P="$BASE/pipeline/scripts"
export TRACK2_LAUNCH_NAME='v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821'
export TRACK2_LAUNCH_RELEASE="$BASE/artifacts/strict_track2_joint_augmentation_20260810/v303_v301_batched_gates_seed1482_20260821"
export TRACK2_LAUNCH_MODEL_VERSION='track2-v301-batched-terminal-frame-mirror-v295'
export TRACK2_LAUNCH_PREPARE_SCRIPT='prepare_v308_v301_rtx5090_official_rl.py'
export TRACK2_LAUNCH_SERVICE_SCRIPT='restart_v308_rtx5090_services.sh'
export TRACK2_LAUNCH_ACCEPT_MARKER='V308_TRAINING_ACCEPTED'

screen -S v308_cpu_governor -X quit >/dev/null 2>&1 || true
screen -dmS v308_cpu_governor bash "$P/govern_v308_cpu_affinity.sh"
exec bash "$P/launch_v300_v295_official_rl.sh"
