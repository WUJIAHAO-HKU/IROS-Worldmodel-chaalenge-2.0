#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge'; O="$B/artifacts/strict_track2_official_20260810"; N='v255_v254_delta_regime_service_seed1457_20260819'
test ! -e "$O/run_registry/$N"; test ! -e "$B/artifacts/strict_track2_joint_augmentation_20260810/$N"
screen -dmS v255_audit bash -lc "bash '$B/pipeline/scripts/launch_v255_service_audit.sh' >> '$O/run_registry/$N.console.log' 2>&1"
sleep 2; screen -ls|grep '[.]v255_audit'; echo V255_STARTED
