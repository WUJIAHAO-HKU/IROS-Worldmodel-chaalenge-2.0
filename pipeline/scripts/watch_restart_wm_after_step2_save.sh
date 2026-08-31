#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/IROS_WAM_2.0_challenge
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME=v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
LOG="$REG/wm_step2_safe_pause.log"
MAIN_PID=34418
RESTART_BELOW_BYTES=$((48 * 1024 * 1024 * 1024))
STEP2_DIR="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_2"

printf '%s watcher_started threshold_bytes=%s\n' "$(date -u +%FT%TZ)" "$RESTART_BELOW_BYTES" >>"$LOG"
while kill -0 "$MAIN_PID" 2>/dev/null; do
  current=$(cat /sys/fs/cgroup/memory.current)
  if (( current < RESTART_BELOW_BYTES )) || [[ -d "$STEP2_DIR" ]]; then
    printf '%s restart_trigger memory_bytes=%s step2_dir=%s\n' \
      "$(date -u +%FT%TZ)" "$current" "$([[ -d "$STEP2_DIR" ]] && echo true || echo false)" >>"$LOG"
    if TRACK2_CPUSET=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      bash "$BASE/pipeline/scripts/restart_v271_v274_services.sh" start >>"$LOG" 2>&1; then
      printf '%s services_ready\n' "$(date -u +%FT%TZ)" >>"$LOG"
      exit 0
    fi
    printf '%s restart_failed_retrying\n' "$(date -u +%FT%TZ)" >>"$LOG"
  fi
  sleep 2
done
printf '%s main_process_exited_before_restart\n' "$(date -u +%FT%TZ)" >>"$LOG"
exit 1
