#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821'
REG="$OFF/run_registry/$NAME"
LOG="$REG/v308_cpu_affinity_governor.log"
CAP='0-21'

for _ in $(seq 1 120); do
  [[ -s "$REG/preregistration.json" ]] && break
  screen -ls 2>/dev/null | grep -q '[.]v308_v301_rtx5090_official_rl' || exit 0
  sleep 1
done
[[ -s "$REG/preregistration.json" ]] || exit 3

owned_pids() {
  {
    pgrep -f '^ray::' || true
    pgrep -f "$NAME" || true
    pgrep -f 'wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005' || true
    pgrep -f 'python pipeline/scripts/serve.py' || true
  } | sort -nu
}

apply_cap() {
  local pid
  while read -r pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      taskset -apc "$CAP" "$pid" >/dev/null 2>&1 || true
      renice 5 -p "$pid" >/dev/null 2>&1 || true
    fi
  done < <(owned_pids)
}

printf '%s governor_started affinity=%s quota_cores=25 reserved_control_cores=3\n' \
  "$(date -u +%FT%TZ)" "$CAP" >>"$LOG"
while screen -ls 2>/dev/null | grep -q '[.]v308_v301_rtx5090_official_rl'; do
  apply_cap
  printf '%s affinity_reapplied=%s\n' "$(date -u +%FT%TZ)" "$CAP" >>"$LOG"
  sleep 20
done
printf '%s outer_training_screen_exited\n' "$(date -u +%FT%TZ)" >>"$LOG"
