#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge'; O="$B/artifacts/strict_track2_official_20260810"; N='v252_v250_specific_terminal_long128_seed1455_20260819'
test ! -e "$O/run_registry/$N"
test ! -e "$B/artifacts/strict_track2_joint_augmentation_20260810/$N"
bash -n "$B/pipeline/scripts/launch_v252_long_gate.sh"
screen -dmS v252_long bash -lc "bash '$B/pipeline/scripts/launch_v252_long_gate.sh' >> '$O/run_registry/$N.console.log' 2>&1"
sleep 2
screen -ls | grep '[.]v252_long'
echo V252_LONG_STARTED
