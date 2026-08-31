#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_CAPTURE_RUN='v259_v169step5_rightterminal_chunk8_sft128_h8_step1_lr5e6_seed1461_20260819'
export TRACK2_CAPTURE_NAME='v260_v259_public_batch00_action_capture_20260819'
export TRACK2_CAPTURE_VARIANT='v260_v259_same_batch00_action_capture'
export TRACK2_CAPTURE_STEP=1
export TRACK2_CAPTURE_TRAIN_REPORT='extra_sft_acceptance.json'
export TRACK2_CAPTURE_PRIOR_GATE_REPORT='public_right_r0_acceptance.json'
export TRACK2_CAPTURE_PRIOR_VARIANT='v259_rightterminal_chunk8_sft128_seed1461'
exec bash "$BASE/pipeline/scripts/launch_v224_v223_action_capture.sh"
