#!/usr/bin/env bash
set -euo pipefail

cd '/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF='artifacts/strict_track2_official_20260810'
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
for _ in $(seq 1 240); do
  if [[ -f "$REG/V318_GLOBAL_STEP6_GATE_PASSED" ]]; then
    echo V318_GLOBAL_STEP6_GATE_PASSED
    sed -n '1,320p' "$RUN/audit/v318_global_step6_continuation_gate.json"
    exit 0
  fi
  if [[ -f "$REG/V318_GLOBAL_STEP6_GATE_FAILED" ]]; then
    echo V318_GLOBAL_STEP6_GATE_FAILED
    sed -n '1,320p' "$RUN/audit/v318_global_step6_continuation_gate.json"
    exit 2
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_resume_from_step3'; then
    echo V318_RESUME_SCREEN_GONE
    tail -80 "$REG/step3_resume_attempts.log" 2>/dev/null || true
    exit 3
  fi
  if awk '/^(max|oom|oom_kill) / && $2 != 0 {bad=1} END {exit !bad}' /sys/fs/cgroup/memory.events 2>/dev/null; then
    echo RESOURCE_SAFETY_FAILURE
    cat /sys/fs/cgroup/memory.events
    exit 6
  fi
  sleep 45
done
exit 5
