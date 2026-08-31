#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
SCREEN='v285b_motion_right'
LOG="$BASE/artifacts/strict_track2_official_20260810/run_registry/v285b_launcher_console.log"

if screen -list | grep -q "[.]$SCREEN"; then
  echo "$SCREEN already running" >&2
  exit 2
fi
screen -dmS "$SCREEN" bash -lc "exec bash '$BASE/pipeline/scripts/launch_v285_motion_weighted_right_sft.sh' >'$LOG' 2>&1"
screen -list | grep "[.]$SCREEN"
