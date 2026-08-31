#!/usr/bin/env bash
set -euo pipefail

BASE="/root/autodl-tmp/IROS_WAM_2.0 challenge"
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
NAME="v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"

for _ in $(seq 1 5760); do
  if [[ -f "$RUN/audit/V304_TRAINING_ACCEPTED" ]]; then
    while screen -ls 2>/dev/null | grep -q '[.]v304_v301_official_rl'; do
      sleep 5
    done
    bash "$P/launch_v305_v304_batch00_gate.sh" \
      > "$REG/v305_batch00_pipeline.log" 2>&1 || {
        rc=$?
        if [[ -f "$RUN/audit/PUBLIC_BATCH00_REJECTED" ]]; then
          echo V305_BATCH00_REJECTED_NO_LATER_PUBLIC_OR_FINAL128_ACCESSED >&2
        fi
        exit "$rc"
      }
    test -f "$RUN/audit/PUBLIC_BATCH00_PASSED"
    echo V304_TO_V305_PIPELINE_COMPLETE
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v304_v301_official_rl'; then
    echo V304_STOPPED_WITHOUT_TRAINING_ACCEPTANCE >&2
    exit 3
  fi
  sleep 30
done

echo V304_TO_V305_WATCH_TIMEOUT >&2
exit 4
