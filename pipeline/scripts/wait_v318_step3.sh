#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN='artifacts/strict_track2_official_20260810/runs/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
REG='artifacts/strict_track2_official_20260810/run_registry/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'

for _ in $(seq 1 240); do
  if [[ -s "$RUN/audit/v318_training_prefix_step3.json" ]]; then
    echo STEP3_READY
    sed -n '1,360p' "$RUN/audit/v318_training_prefix_step3.json"
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_v317_official_rl'; then
    echo TRAINING_SCREEN_GONE
    tail -40 "$REG/recovery_attempts.log" 2>/dev/null || true
    tail -60 "$RUN/launcher.log" 2>/dev/null || true
    exit 3
  fi
  if grep -aqE 'Traceback|OutOfMemory|CUDA out of memory|HTTP.*(500|502|503|504)' "$RUN/launcher.log"; then
    echo TRAINING_ERROR
    grep -aE 'Traceback|OutOfMemory|CUDA out of memory|HTTP.*(500|502|503|504)' "$RUN/launcher.log" | tail -30
    exit 4
  fi
  if awk '/^(max|oom|oom_kill) / && $2 != 0 {bad=1} END {exit !bad}' /sys/fs/cgroup/memory.events 2>/dev/null; then
    echo RESOURCE_SAFETY_FAILURE
    cat /sys/fs/cgroup/memory.events
    exit 6
  fi
  sleep 45
done

echo STEP3_WAIT_TIMEOUT
exit 5
