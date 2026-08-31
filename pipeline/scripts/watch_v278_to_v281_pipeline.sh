#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0_challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"

for _ in $(seq 1 5760); do
  if [[ -f "$RUN/audit/V278_RETRY2_TRAINING_ACCEPTED" ]]; then
    while screen -ls 2>/dev/null | grep -q '[.]v278_retry2_rl'; do sleep 5; done
    bash "$BASE/pipeline/scripts/launch_v279_v278_batch00_gate.sh" \
      >"$REG/v279_batch00_pipeline.log" 2>&1 || {
        rc=$?
        if [[ -f "$RUN/audit/PUBLIC_BATCH00_REJECTED" ]]; then
          echo V279_BATCH00_REJECTED_NO_BATCH01_OR_FINAL128_ACCESSED >&2
        fi
        exit "$rc"
      }
    test -f "$RUN/audit/PUBLIC_BATCH00_PASSED"
    bash "$BASE/pipeline/scripts/launch_v280_v278_public112_freeze_final.sh" \
      >"$REG/v280_public112_freeze_final_pipeline.log" 2>&1
    exit $?
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v278_retry2_rl'; then
    echo V278_STOPPED_WITHOUT_TRAINING_ACCEPTANCE >&2
    exit 3
  fi
  sleep 30
done
echo V278_TO_V281_WATCH_TIMEOUT >&2
exit 4
