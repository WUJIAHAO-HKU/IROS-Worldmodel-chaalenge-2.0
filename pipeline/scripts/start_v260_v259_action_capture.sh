#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v260_v259_public_batch00_action_capture_20260819'
test ! -e "$OFF/run_registry/$NAME"
screen -dmS v260_v259_action_capture bash -lc \
  "bash '$BASE/pipeline/scripts/launch_v260_v259_action_capture.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v260_v259_action_capture
echo V260_V259_ACTION_CAPTURE_STARTED
