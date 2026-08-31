#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v267_v266_public_batch00_action_capture_20260819'

test ! -e "$OFF/run_registry/$NAME"
screen -dmS v267_v266_action_capture bash -lc \
  "bash '$BASE/pipeline/scripts/launch_v267_v266_action_capture.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v267_v266_action_capture
echo V267_V266_ACTION_CAPTURE_STARTED
