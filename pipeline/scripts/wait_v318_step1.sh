#!/usr/bin/env bash
set -euo pipefail
cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN='artifacts/strict_track2_official_20260810/runs/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
REG='artifacts/strict_track2_official_20260810/run_registry/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
for _ in $(seq 1 50); do
  if [[ -s "$RUN/audit/v318_training_prefix_step1.json" ]]; then
    echo STEP1_READY
    sed -n '1,280p' "$RUN/audit/v318_training_prefix_step1.json"
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_v317_official_rl'; then
    echo TRAINING_SCREEN_GONE
    tail -30 "$REG/recovery_attempts.log"
    exit 3
  fi
  if grep -aqE 'Traceback|OutOfMemory|CUDA out of memory|HTTP.*(500|502|503|504)' "$RUN/launcher.log"; then
    echo TRAINING_ERROR
    grep -aE 'Traceback|OutOfMemory|CUDA out of memory|HTTP.*(500|502|503|504)' "$RUN/launcher.log" | tail -20
    exit 4
  fi
  sleep 45
done
echo STEP1_WAIT_TIMEOUT
exit 5
