#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
export TRACK2_GATE_NAME='v229_v228_terminal_rightsft_continue_h8_step64_seed1429_20260818'
export TRACK2_GATE_VARIANT='v229_v228_terminal_rightsft_continue_step64_seed1429'
export TRACK2_GATE_TRAIN_MARKER='V229_TERMINAL_RIGHT_SFT_RETRY_ACCEPTED'
exec bash "$BASE/pipeline/scripts/launch_v228_public_right_gate.sh"
