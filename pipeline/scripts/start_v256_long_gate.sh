#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge';O="$B/artifacts/strict_track2_official_20260810";N='v256_v254_delta_regime_long128_seed1458_20260819';test ! -e "$O/run_registry/$N";test ! -e "$B/artifacts/strict_track2_joint_augmentation_20260810/$N";bash -n "$B/pipeline/scripts/launch_v256_long_gate.sh";screen -dmS v256_long bash -lc "bash '$B/pipeline/scripts/launch_v256_long_gate.sh' >> '$O/run_registry/$N.console.log' 2>&1";sleep 2;screen -ls|grep '[.]v256_long';echo V256_LONG_STARTED
