#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/IROS_WAM_2.0_challenge
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME=v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
TARGET_STEP="${1:-2}"
LOG="$REG/wm_step${TARGET_STEP}_safe_pause.log"
TARGET_DIR="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${TARGET_STEP}"
MODEL_FILE="$TARGET_DIR/actor/model_state_dict/full_weights.pt"
OPTIMIZER_FILE="$TARGET_DIR/actor/optimizer_recovery.pt"
TRAIN_PATTERN="train_embodied_agent.py.*$NAME"
MAX_STEP="${TRACK2_MAX_STEPS:-10}"
SAVE_PRESSURE_GRACE_SECONDS="${SAVE_PRESSURE_GRACE_SECONDS:-20}"
EXIT_CHECKPOINT_GRACE_SECONDS="${EXIT_CHECKPOINT_GRACE_SECONDS:-180}"

model_zip_ready() {
  [[ -s "$MODEL_FILE" ]] || return 1
  python - "$MODEL_FILE" <<'PY'
import sys, zipfile
raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
}

optimizer_zip_ready() {
  [[ -s "$OPTIMIZER_FILE" ]] || return 1
  python - "$OPTIMIZER_FILE" <<'PY'
import sys, zipfile
raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
}

checkpoint_ready() {
  model_zip_ready || return 1
  if (( TARGET_STEP < MAX_STEP )); then
    optimizer_zip_ready || return 1
  fi
}

find_saving_actor() {
  pgrep -f '^ray::EmbodiedFSDPActor.save_checkpoint' | head -1 || true
}

find_rollout_worker() {
  pgrep -f '^ray::MultiStepRolloutWorker' | head -1 || true
}

release_idle_rollout_for_save() {
  local rollout_pid=$1 cmd
  [[ -r "/proc/$rollout_pid/cmdline" ]] || return 0
  cmd=$(tr '\0' ' ' <"/proc/$rollout_pid/cmdline")
  [[ "$cmd" == *MultiStepRolloutWorker* ]] || return 1
  printf '%s releasing_idle_rollout pid=%s signal=TERM\n' \
    "$(date -u +%FT%TZ)" "$rollout_pid" >>"$LOG"
  kill -TERM "$rollout_pid" 2>/dev/null || true
  for _ in 1 2; do
    sleep 5
    [[ -d "/proc/$rollout_pid" ]] || return 0
  done
  if [[ -r "/proc/$rollout_pid/cmdline" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/$rollout_pid/cmdline")
    if [[ "$cmd" == *MultiStepRolloutWorker* ]]; then
      printf '%s releasing_idle_rollout pid=%s signal=KILL\n' \
        "$(date -u +%FT%TZ)" "$rollout_pid" >>"$LOG"
      kill -KILL "$rollout_pid" 2>/dev/null || true
    fi
  fi
}

printf '%s watcher_started restart_condition=checkpoint_complete max_step=%s pressure_grace_seconds=%s\n' \
  "$(date -u +%FT%TZ)" "$MAX_STEP" "$SAVE_PRESSURE_GRACE_SECONDS" >>"$LOG"
main_pid=$(pgrep -f "$TRAIN_PATTERN" | head -1 || true)
if [[ -z "$main_pid" ]]; then
  printf '%s training_process_not_found\n' "$(date -u +%FT%TZ)" >>"$LOG"
  exit 1
fi
printf '%s training_pid=%s target_step=%s\n' "$(date -u +%FT%TZ)" "$main_pid" "$TARGET_STEP" >>"$LOG"
save_pressure_since=0
while true; do
  current=$(cat /sys/fs/cgroup/memory.current)
  if checkpoint_ready; then
    printf '%s restart_trigger memory_bytes=%s checkpoint_complete=true\n' \
      "$(date -u +%FT%TZ)" "$current" >>"$LOG"
    if TRACK2_CPUSET=0 TRACK2_SERVICE_REG="$REG" \
      WAM_IMAGE_CODEC_WORKERS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      bash "$BASE/pipeline/scripts/restart_v271_v274_services.sh" start >>"$LOG" 2>&1; then
      printf '%s services_ready\n' "$(date -u +%FT%TZ)" >>"$LOG"
      exit 0
    fi
    printf '%s restart_failed_retrying\n' "$(date -u +%FT%TZ)" >>"$LOG"
  fi

  saving_actor=$(find_saving_actor)
  rollout_pid=$(find_rollout_worker)
  memory_high=$(cat /sys/fs/cgroup/memory.high)
  if (( TARGET_STEP < MAX_STEP )) && \
     [[ -n "$saving_actor" && -n "$rollout_pid" && "$memory_high" != max ]] && \
     (( current >= memory_high - 1073741824 )); then
    now=$(date +%s)
    if (( save_pressure_since == 0 )); then
      save_pressure_since=$now
      printf '%s save_memory_pressure actor_pid=%s rollout_pid=%s memory_bytes=%s high_bytes=%s\n' \
        "$(date -u +%FT%TZ)" "$saving_actor" "$rollout_pid" "$current" "$memory_high" >>"$LOG"
    elif (( now - save_pressure_since >= SAVE_PRESSURE_GRACE_SECONDS )); then
      release_idle_rollout_for_save "$rollout_pid"
      save_pressure_since=0
    fi
  else
    save_pressure_since=0
  fi

  if ! kill -0 "$main_pid" 2>/dev/null; then
    # Ray can report the deliberately released rollout worker before the
    # asynchronous checkpoint writer has flushed its ZIP central directory.
    # Keep checking for a bounded grace period, then restore services from the
    # completed checkpoint even though the original driver has exited.
    exit_deadline=$(( $(date +%s) + EXIT_CHECKPOINT_GRACE_SECONDS ))
    while (( $(date +%s) < exit_deadline )); do
      if checkpoint_ready; then
        printf '%s checkpoint_completed_after_main_exit grace_seconds=%s\n' \
          "$(date -u +%FT%TZ)" "$EXIT_CHECKPOINT_GRACE_SECONDS" >>"$LOG"
        break
      fi
      sleep 2
    done
    checkpoint_ready && continue
    printf '%s main_process_exited_before_restart\n' "$(date -u +%FT%TZ)" >>"$LOG"
    exit 1
  fi
  sleep 2
done
