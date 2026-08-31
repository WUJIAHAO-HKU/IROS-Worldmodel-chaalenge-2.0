#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_GATE_NAME='v232_v228_transition_rightsft_joint2_h8_step64_seed1432_20260818'
export TRACK2_GATE_VARIANT='v233_v232_transition_rightsft_joint2_step64_seed1432'
export TRACK2_GATE_TRAIN_MARKER='V232_TERMINAL_RIGHT_SFT_RETRY_ACCEPTED'
exec bash "$BASE/pipeline/scripts/launch_v228_public_right_gate.sh"
