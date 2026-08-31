#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_CAPTURE_RUN='v229_v228_terminal_rightsft_continue_h8_step64_seed1429_20260818'
export TRACK2_CAPTURE_NAME='v230_v229_public_batch00_action_capture_20260818'
export TRACK2_CAPTURE_VARIANT='v230_v229_same_batch00_action_capture'
export TRACK2_CAPTURE_STEP=64
export TRACK2_CAPTURE_TRAIN_REPORT='terminal_right_sft_retry_acceptance.json'
export TRACK2_CAPTURE_PRIOR_VARIANT='v229_v228_terminal_rightsft_continue_step64_seed1429'
exec bash "$BASE/pipeline/scripts/launch_v224_v223_action_capture.sh"
