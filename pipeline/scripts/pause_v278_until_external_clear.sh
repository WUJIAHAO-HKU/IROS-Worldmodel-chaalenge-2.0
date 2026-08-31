#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/IROS_WAM_2.0_challenge
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME=v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
STEP2="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_2/actor"
MODEL="$STEP2/model_state_dict/full_weights.pt"
OPT="$STEP2/optimizer_recovery.pt"
LOG="$REG/controlled_pause_step2.log"

"$PY" - "$MODEL" "$OPT" <<'PY'
import sys, zipfile
from pathlib import Path
model, opt = map(Path, sys.argv[1:])
assert model.stat().st_size == 8529316588
assert opt.stat().st_size == 3253939504
assert zipfile.is_zipfile(model) and zipfile.is_zipfile(opt)
PY

printf '%s controlled_pause_begin checkpoint=global_step_2\n' "$(date -u +%FT%TZ)" >>"$LOG"
for name in v278_retry2_rl wm_v271_v274_bridge wm_v271_v274_gpu govern_v278_cpu watch_v278_retry2_health; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
pkill -TERM -f "train_embodied_agent.py.*$NAME" >/dev/null 2>&1 || true
sleep 5
timeout 120 "$PY" -m ray.scripts.scripts stop --force >>"$LOG" 2>&1 || true
printf '%s owned_training_stopped\n' "$(date -u +%FT%TZ)" >>"$LOG"

clear_count=0
while (( clear_count < 3 )); do
  external=false
  if timeout 10 pgrep -f '/root/autodl-tmp/Unity/.*(Unity|GradleDaemon)' >/dev/null 2>&1; then
    external=true
  fi
  if [[ "$external" == false ]]; then
    clear_count=$((clear_count + 1))
  else
    clear_count=0
  fi
  printf '%s external=%s consecutive_clear=%s memory_bytes=%s\n' \
    "$(date -u +%FT%TZ)" "$external" "$clear_count" "$(cat /sys/fs/cgroup/memory.current)" >>"$LOG"
  (( clear_count < 3 )) && sleep 30
done

stamp=$(date -u +%Y%m%dT%H%M%SZ)
supervisor_log="$REG/supervisor_resume_step2_${stamp}.log"
screen -L -Logfile "$supervisor_log" -dmS v278_retry2_rl \
  env TRACK2_OPERATOR_RESTART=1 TRACK2_MODEL_ONLY_RECOVERY=1 TRACK2_CPUSET=0-2 \
  TRACK2_OMP_NUM_THREADS=1 TRACK2_MKL_NUM_THREADS=1 TRACK2_OPENBLAS_NUM_THREADS=1 \
  TRACK2_NUMEXPR_NUM_THREADS=1 TRACK2_RAYON_NUM_THREADS=1 \
  TRACK2_TF_NUM_INTRAOP_THREADS=1 TRACK2_TF_NUM_INTEROP_THREADS=1 \
  bash "$BASE/pipeline/scripts/launch_v278_retry2_v271_resumable_rl.sh"
screen -S govern_v278_cpu -X quit >/dev/null 2>&1 || true
screen -dmS govern_v278_cpu bash "$BASE/pipeline/scripts/govern_v278_cpu_affinity.sh"
screen -S watch_v278_retry2_health -X quit >/dev/null 2>&1 || true
screen -dmS watch_v278_retry2_health env TRACK2_CPU_CAP_CORES=3 \
  bash "$BASE/pipeline/scripts/watch_v278_retry2_training_health.sh"
screen -S watch_v278_to_v281 -X quit >/dev/null 2>&1 || true
screen -dmS watch_v278_to_v281 bash "$BASE/pipeline/scripts/watch_v278_to_v281_pipeline.sh"
printf '%s resume_launched checkpoint=global_step_2 supervisor_log=%s\n' \
  "$(date -u +%FT%TZ)" "$supervisor_log" >>"$LOG"
