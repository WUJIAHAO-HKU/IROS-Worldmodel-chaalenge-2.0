#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/IROS_WAM_2.0_challenge
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME=v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
TRAIN_LOG="$RUN/launcher.log"
CHECKPOINT_ROOT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints"
START_STEP="${1:-5}"
END_STEP="${2:-10}"
SAVE_INTERVAL="${3:-3}"
INITIAL_CHECKPOINT_STEP="${4:-4}"
EVENT_TIMEOUT_SECONDS="${EVENT_TIMEOUT_SECONDS:-7200}"
LOG="$REG/wm_save_cycle_manager.log"
TRAIN_PATTERN="train_embodied_agent.py.*$NAME"

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"
}

training_alive() {
  pgrep -f "$TRAIN_PATTERN" >/dev/null
}

wait_for_initial_checkpoint() {
  local initial_model="$CHECKPOINT_ROOT/global_step_${INITIAL_CHECKPOINT_STEP}/actor/model_state_dict/full_weights.pt"
  local initial_optimizer="$CHECKPOINT_ROOT/global_step_${INITIAL_CHECKPOINT_STEP}/actor/optimizer_recovery.pt"
  while training_alive; do
    [[ -s "$initial_model" && -s "$initial_optimizer" ]] && return 0
    sleep 2
  done
  return 1
}

wait_for_rollout_completion() {
  set +e
  timeout "$EVENT_TIMEOUT_SECONDS" \
    tail -n0 -F "$TRAIN_LOG" | \
    grep -a -m1 'Generating Rollout Epochs: 100%' >/dev/null
  local pipeline_status=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipeline_status[1]:-1}" -eq 0 ]]
}

wait_for_services_ready() {
  local target_step=$1 safe_pause_log="$REG/wm_step${target_step}_safe_pause.log"
  while training_alive; do
    if [[ -f "$safe_pause_log" ]] && \
       grep -q 'services_ready' "$safe_pause_log"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

if (( START_STEP < 3 || END_STEP < START_STEP || SAVE_INTERVAL < 1 )); then
  printf 'invalid step range: %s..%s\n' "$START_STEP" "$END_STEP" >&2
  exit 2
fi

log "manager_started steps=${START_STEP}-${END_STEP} save_interval=${SAVE_INTERVAL} initial_checkpoint_step=${INITIAL_CHECKPOINT_STEP} timeout_seconds=${EVENT_TIMEOUT_SECONDS}"
if ! wait_for_initial_checkpoint; then
  log "training_exited_before_initial_checkpoint step=${INITIAL_CHECKPOINT_STEP}"
  exit 1
fi
for target_step in $(seq "$START_STEP" "$END_STEP"); do
  log "waiting_for_rollout_completion target_step=${target_step}"
  if ! wait_for_rollout_completion; then
    log "rollout_wait_failed target_step=${target_step}"
    exit 1
  fi

  if (( target_step != END_STEP && target_step % SAVE_INTERVAL != 0 )); then
    log "rollout_complete target_step=${target_step} checkpoint_skipped=true"
    continue
  fi

  log "rollout_complete target_step=${target_step} stopping_world_models"
  screen -S "watch_restart_wm_step${target_step}" -X quit >/dev/null 2>&1 || true
  screen -dmS "watch_restart_wm_step${target_step}" \
    bash "$BASE/pipeline/scripts/watch_restart_wm_after_step_save.sh" "$target_step"
  bash "$BASE/pipeline/scripts/restart_v271_v274_services.sh" stop >>"$LOG" 2>&1

  if ! wait_for_services_ready "$target_step"; then
    log "services_not_restored target_step=${target_step}"
    exit 1
  fi
  log "checkpoint_cycle_complete target_step=${target_step}"
done
log "manager_complete steps=${START_STEP}-${END_STEP}"
