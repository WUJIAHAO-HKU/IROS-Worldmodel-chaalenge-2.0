#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
RUN="$OFF/runs/v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819"
for _ in $(seq 1 5760);do
  if [[ -f "$RUN/audit/PUBLIC_BATCH00_REJECTED" ]];then
    echo V275_BATCH00_REJECTED_NO_LATER_DATA_ACCESSED >&2;exit 4
  fi
  if [[ -f "$RUN/audit/PUBLIC_BATCH00_PASSED" ]];then
    while screen -ls 2>/dev/null|grep -q '[.]v275_v274r1_b00';do sleep 5;done
    test ! -e "$RUN/audit/PUBLIC112_PASSED"
    test ! -e "$RUN/audit/PUBLIC112_REJECTED"
    screen -ls 2>/dev/null|grep -q '[.]v276_v274r1_public112_final'&&exit 9||true
    screen -dmS v276_v274r1_public112_final bash -lc \
      "bash \"$BASE/pipeline/scripts/launch_v276_v274_public112_freeze_final.sh\" >\"$OFF/run_registry/v276_v274r1_public112_final.log\" 2>&1"
    echo V276_PUBLIC112_FREEZE_FINAL_LAUNCHED;exit 0
  fi
  sleep 30
done
echo V275_TO_V276_WATCH_TIMEOUT >&2
exit 5
