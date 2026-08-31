#!/usr/bin/env bash
set -euo pipefail

BASE="/root/autodl-tmp/IROS_WAM_2.0 challenge"
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
NAME="v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"

for _ in $(seq 1 5760); do
  if [[ -f "$RUN/audit/PUBLIC_BATCH00_PASSED" ]]; then
    while screen -ls 2>/dev/null | grep -q '[.]v304_to_v305_pipeline'; do
      sleep 5
    done
    bash "$P/launch_v306_v304_public112_freeze_final.sh" \
      > "$REG/v306_public112_freeze_final_pipeline.log" 2>&1
    exit $?
  fi
  if [[ -f "$RUN/audit/PUBLIC_BATCH00_REJECTED" ]]; then
    echo V305_REJECTED_NO_PUBLIC112_OR_FINAL128_ACCESSED >&2
    exit 2
  fi
  if ! screen -ls 2>/dev/null | grep -qE \
    '[.]v304_v301_official_rl|[.]v304_to_v305_pipeline'; then
    echo V304_V305_PIPELINE_STOPPED_WITHOUT_BATCH00_DECISION >&2
    exit 3
  fi
  sleep 30
done

echo V305_TO_V307_WATCH_TIMEOUT >&2
exit 4
