#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_CAPTURE_RUN='v261_v169step5_fullright_chunk8_sft512_grip8_inactive1_seed1462_20260819'
export TRACK2_CAPTURE_NAME='v262_v261_public_batch00_action_capture_20260819'
export TRACK2_CAPTURE_VARIANT='v262_v261_same_batch00_action_capture'
export TRACK2_CAPTURE_STEP=1
export TRACK2_CAPTURE_TRAIN_REPORT='extra_sft_acceptance.json'
export TRACK2_CAPTURE_PRIOR_GATE_REPORT='public_right_r0_acceptance.json'
export TRACK2_CAPTURE_PRIOR_VARIANT='v261_fullright_chunk8_sft512_grip8_seed1462'

exec bash "$BASE/pipeline/scripts/launch_v224_v223_action_capture.sh"
