#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
O="$ROOT/artifacts/strict_track2_official_20260810"
GATE="$J/v273_v271_corrected_alpha_long_gate_seed1470_20260819"
RL_NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_20260819'
for _ in $(seq 1 720); do
  if [[ -f "$GATE/V273_LONG_GATE_PASSED" ]]; then
    test -f "$GATE/AUDIT_COMPLETE"
    test ! -e "$O/runs/$RL_NAME"
    test ! -e "$O/run_registry/$RL_NAME"
    screen -ls | grep -q '[.]v274_v271_rl' && exit 9 || true
    screen -dmS v274_v271_rl bash -lc "bash \"$ROOT/pipeline/scripts/launch_v274_v271_fullbudget_rl.sh\" >\"$O/run_registry/v274_v271_rl_launcher_pending.log\" 2>&1"
    echo V274_RL_LAUNCHED
    exit 0
  fi
  if ! screen -ls | grep -q '[.]v273_v271_gate'; then
    echo V273_GATE_STOPPED_WITHOUT_PASS >&2
    exit 3
  fi
  sleep 30
done
echo V273_GATE_WATCH_TIMEOUT >&2
exit 4
