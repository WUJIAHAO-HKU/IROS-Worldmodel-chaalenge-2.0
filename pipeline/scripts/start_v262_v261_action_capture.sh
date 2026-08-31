#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v262_v261_public_batch00_action_capture_20260819'

test ! -e "$OFF/run_registry/$NAME"
screen -dmS v262_v261_action_capture bash -lc \
  "bash '$BASE/pipeline/scripts/launch_v262_v261_action_capture.sh' >> '$OFF/run_registry/$NAME.console.log' 2>&1"
sleep 2
screen -ls | grep -q v262_v261_action_capture
echo V262_V261_ACTION_CAPTURE_STARTED
