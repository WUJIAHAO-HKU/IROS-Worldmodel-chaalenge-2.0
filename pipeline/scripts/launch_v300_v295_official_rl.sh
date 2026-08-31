#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'; OFF="$ROOT/artifacts/strict_track2_official_20260810"; JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME="${TRACK2_LAUNCH_NAME:-v300_v295_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821}"; RUN="$OFF/runs/$NAME"; REG="$OFF/run_registry/$NAME"; P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'; RELEASE="${TRACK2_LAUNCH_RELEASE:-$JOINT/v299_v295_corrected_numeric_long_gate_seed1481_20260821}"; MANIFEST="$RELEASE/release_registration.json"
MODEL_VERSION="${TRACK2_LAUNCH_MODEL_VERSION:-track2-v295-terminal-frame-preserving-mirror-v271}"; EXPERIMENT='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'; PREPARE_SCRIPT="${TRACK2_LAUNCH_PREPARE_SCRIPT:-prepare_v300_v295_official_rl.py}"; SERVICE_SCRIPT="${TRACK2_LAUNCH_SERVICE_SCRIPT:-restart_v295_services.sh}"; ACCEPT_MARKER="${TRACK2_LAUNCH_ACCEPT_MARKER:-V300_TRAINING_ACCEPTED}"
CHECKPOINT="$RUN/$EXPERIMENT/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"; REFERENCE="$OFF/immutable_reference_policy_state/pi05_official_rank0.pt"
SERVICE_CPUSET="${TRACK2_SERVICE_CPUSET:-0-15}"; RL_CPUSET="${TRACK2_RL_CPUSET:-16-47}"
restore(){ "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; bash "$P/$SERVICE_SCRIPT" stop >/dev/null 2>&1 || true; TRACK2_CPUSET=0-15 bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_after.log" 2>&1 || true; }
trap restore EXIT
if [[ ! -e "$REG" && ! -e "$RUN" ]]; then
  "$PY" "$P/$PREPARE_SCRIPT" >"$OFF/run_registry/$NAME.prepare.tmp.log" 2>&1
  mv "$OFF/run_registry/$NAME.prepare.tmp.log" "$REG/prepare.log"
elif [[ -s "$REG/preregistration.json" && ! -e "$RUN" ]]; then
  echo REUSING_FROZEN_PREREGISTRATION >>"$REG/prepare.log"
else
  echo 'refusing ambiguous existing v300 state' >&2; exit 3
fi
TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET="$SERVICE_CPUSET" TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/$SERVICE_SCRIPT" start >"$REG/restart_candidate_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"
export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN" TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005' TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=10 TRACK2_SAVE_INTERVAL=3 TRACK2_KEEP_LAST_CHECKPOINTS=2
export TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1471 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=2e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200 TRACK2_ROLLOUT_EPOCH=4 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=3200
export TRACK2_REFERENCE_STATE_STORAGE=disk TRACK2_REFERENCE_STATE_PATH_OVERRIDE="$REFERENCE"
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true
export TRACK2_CATCH_SYSTEM_FAILURE=0 TRACK2_ACTOR_OFFLOAD=true TRACK2_ENABLE_SFT_CO_TRAIN=false TOKENIZERS_PARALLELISM=false
export TRACK2_CPUSET="$RL_CPUSET" TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TRACK2_RAY_SYSTEM_CONFIG_JSON='{"grpc_client_keepalive_timeout_ms":1800000,"grpc_keepalive_timeout_ms":1800000,"health_check_timeout_ms":1800000,"health_check_failure_threshold":10}'
completed=false
for attempt in $(seq 1 24); do
  printf 'attempt=%s start_utc=%s start_step=%s\n' "$attempt" "$(date -u +%FT%TZ)" "${TRACK2_START_STEP:-0}" >>"$REG/recovery_attempts.log"
  set +e
  bash "$P/run_strict_track2_conservative_kl.sh" >"$REG/runner_attempt_${attempt}.screen.log" 2>&1
  rc=$?
  set -e
  cp -f "$RUN/launcher.log" "$REG/launcher_attempt_${attempt}.log" 2>/dev/null || true
  printf 'attempt=%s exit_rc=%s end_utc=%s\n' "$attempt" "$rc" "$(date -u +%FT%TZ)" >>"$REG/recovery_attempts.log"
  if [[ -s "$CHECKPOINT" ]] && "$PY" -c 'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$CHECKPOINT"; then completed=true; break; fi
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  recovery="$REG/recovery_after_attempt_${attempt}.json"
  "$PY" - "$RUN" "$EXPERIMENT" "$recovery" <<'PY'
import json,sys,zipfile
from pathlib import Path
run,experiment,output=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]); valid=[]
for path in (run/experiment/'checkpoints').glob('global_step_*'):
    try: step=int(path.name.rsplit('_',1)[1])
    except ValueError: continue
    model=path/'actor/model_state_dict/full_weights.pt'; optimizer=path/'actor/optimizer_recovery.pt'
    if step<10 and model.is_file() and optimizer.is_file() and zipfile.is_zipfile(model) and zipfile.is_zipfile(optimizer): valid.append((step,model,optimizer))
payload={"valid_steps":[row[0] for row in sorted(valid)]}
if valid:
    step,model,optimizer=max(valid,key=lambda row:row[0]); payload["selected"]={"step":step,"model":str(model),"optimizer":str(optimizer)}
output.write_text(json.dumps(payload,indent=2)+'\n')
PY
  readarray -t fields < <("$PY" - "$recovery" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])).get('selected',{}); print(x.get('step','')); print(x.get('model','')); print(x.get('optimizer',''))
PY
  )
  if [[ -n "${fields[0]:-}" ]]; then
    export TRACK2_START_STEP="${fields[0]}" TRACK2_CKPT_PATH="${fields[1]}" TRACK2_OPTIMIZER_RECOVERY_PATH="${fields[2]}" TRACK2_ALLOW_EXISTING_RUN=true
  else
    export TRACK2_START_STEP=0 TRACK2_CKPT_PATH='' TRACK2_OPTIMIZER_RECOVERY_PATH='' TRACK2_ALLOW_EXISTING_RUN=true
  fi
done
if [[ "$completed" != true ]]; then echo V300_EXHAUSTED_RECOVERY_ATTEMPTS >&2; exit 9; fi
"$PY" "$P/audit_strict_track2_kl_smoke.py" --run "$RUN" --preregistration "$REG/preregistration.json" --output "$RUN/audit/p3_training_acceptance.json" >"$RUN/audit/p3_training_acceptance.log" 2>&1
touch "$RUN/audit/$ACCEPT_MARKER"
echo "${ACCEPT_MARKER}_OFFICIAL_RL_TRAINING_ACCEPTED"
