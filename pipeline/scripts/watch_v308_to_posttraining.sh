#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PIPELINE="$BASE/pipeline/scripts/run_v308_posttraining_pipeline.sh"

for _ in $(seq 1 5760); do
  if [[ -f "$RUN/audit/V308_TRAINING_ACCEPTED" ]]; then
    while screen -ls 2>/dev/null | grep -q '[.]v308_v301_rtx5090_official_rl'; do
      sleep 5
    done
    exec bash "$PIPELINE" >"$REG/v308_posttraining_pipeline.log" 2>&1
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v308_v301_rtx5090_official_rl'; then
    echo V308_STOPPED_WITHOUT_TRAINING_ACCEPTANCE >&2
    exit 3
  fi
  sleep 30
done

echo V308_POSTTRAINING_WATCH_TIMEOUT >&2
exit 4
