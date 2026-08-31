#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/IROS_WAM_2.0_challenge
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME=v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820
REG="$OFF/run_registry/$NAME"
LOG="$REG/cpu_affinity_governor.log"
state=unknown
TRAIN_PATTERN="train_embodied_agent.py.*$NAME"
NORMAL_CAP=0-9
MEMORY_PRESSURE_CAP=0-5
SCENE_PRESSURE_CAP=0-1
UNITY_PRESSURE_CAP=0
REAPPLY_SECONDS=60

owned_pids() {
  {
    pgrep -f '^ray::' || true
    pgrep -f "$NAME" || true
    pgrep -f 'wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005' || true
    pgrep -f 'python pipeline/scripts/serve.py' || true
  } | sort -nu
}

apply_cap() {
  local cap=$1 pid
  while read -r pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      taskset -apc "$cap" "$pid" >/dev/null 2>&1 || true
    fi
  done < <(owned_pids)
}

printf '%s governor_started\n' "$(date -u +%FT%TZ)" >>"$LOG"
main_pid=
last_main_pid=
last_apply_epoch=0
while screen -ls 2>/dev/null | grep -q '[.]v278_retry2_rl'; do
  main_pid=$(pgrep -f "$TRAIN_PATTERN" | head -1 || true)
  if [[ -z "$main_pid" ]]; then
    if [[ -n "$last_main_pid" ]]; then
      printf '%s training_pid_exited=%s waiting_for_retry=true\n' \
        "$(date -u +%FT%TZ)" "$last_main_pid" >>"$LOG"
      last_main_pid=
      state=unknown
    fi
    sleep 5
    continue
  fi
  if [[ "$main_pid" != "$last_main_pid" ]]; then
    printf '%s training_pid=%s\n' "$(date -u +%FT%TZ)" "$main_pid" >>"$LOG"
    last_main_pid=$main_pid
    state=unknown
  fi

  desired=$NORMAL_CAP
  reason=normal_accelerated
  if pgrep -f '/root/autodl-tmp/Unity/.*(Unity|GradleDaemon)' >/dev/null; then
    desired=$UNITY_PRESSURE_CAP
    reason=external_unity_pressure
  elif pgrep -f 'scripts/validate_scene2_production.py' >/dev/null; then
    desired=$SCENE_PRESSURE_CAP
    reason=external_scene2_pressure
  elif [[ -r /sys/fs/cgroup/memory.current && -r /sys/fs/cgroup/memory.high ]]; then
    memory_current=$(< /sys/fs/cgroup/memory.current)
    memory_high=$(< /sys/fs/cgroup/memory.high)
    if [[ "$memory_high" != max ]] && \
       (( memory_current >= memory_high - 1073741824 )); then
      desired=$MEMORY_PRESSURE_CAP
      reason=memory_near_high
    fi
  fi
  now_epoch=$(date +%s)
  if [[ "$desired" != "$state" ]] || \
     (( now_epoch - last_apply_epoch >= REAPPLY_SECONDS )); then
    apply_cap "$desired"
    if [[ "$desired" != "$state" ]]; then
      printf '%s affinity=%s reason=%s\n' "$(date -u +%FT%TZ)" "$desired" "$reason" >>"$LOG"
    fi
    state=$desired
    last_apply_epoch=$now_epoch
  fi
  sleep 10
done
printf '%s outer_training_screen_exited\n' "$(date -u +%FT%TZ)" >>"$LOG"
