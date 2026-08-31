#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_CAPTURE_RUN='v266_v265_physical14_mirror_chunk8_sft2334_b4_lr1e6_seed1466_20260819'
export TRACK2_CAPTURE_NAME='v267_v266_public_batch00_action_capture_20260819'
export TRACK2_CAPTURE_VARIANT='v267_v266_same_batch00_action_capture'
export TRACK2_CAPTURE_STEP=1
export TRACK2_CAPTURE_TRAIN_REPORT='extra_sft_acceptance.json'
export TRACK2_CAPTURE_PRIOR_GATE_REPORT='public_right_r0_acceptance.json'
export TRACK2_CAPTURE_PRIOR_VARIANT='v266_physical14_mirror_epoch1_seed1466'

exec bash "$BASE/pipeline/scripts/launch_v224_v223_action_capture.sh"
