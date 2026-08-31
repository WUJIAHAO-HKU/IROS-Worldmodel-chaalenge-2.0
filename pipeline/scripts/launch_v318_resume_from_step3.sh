#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
EXP='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RELEASE="$JOINT/v317_v315_native_batch_causal_gate_seed1490_20260822"
MANIFEST="$RELEASE/release_registration.json"
AUTH="$RELEASE/audit/expensive_rl_authorization.json"
REFERENCE="$OFF/immutable_reference_policy_state/pi05_official_rank0.pt"
GATE="$RUN/audit/v318_global_step3_continuation_gate.json"
CHECKPOINT="$RUN/$EXP/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"

for path in "$REG/preregistration.json" "$AUTH" "$MANIFEST" "$REFERENCE" "$GATE"; do test -s "$path"; done
"$PY" - "$GATE" "$AUTH" <<'PY'
import json,sys
gate=json.load(open(sys.argv[1])); auth=json.load(open(sys.argv[2]))
assert gate.get("passed") is True and all(gate.get("checks", {}).values())
assert auth.get("passed") is True and auth.get("launch_permission") is True
PY

restore() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  bash "$P/restart_v317_services.sh" stop >/dev/null 2>&1 || true
  TRACK2_CPUSET=0-15 bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_after_v318_resume.log" 2>&1 || true
}
trap restore EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 TRACK2_WAM_RELEASE_CUDA_CACHE=1 \
  bash "$P/restart_v317_services.sh" start >"$REG/restart_v317_before_step3_resume.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q 'track2-v317-batched-sparse-failure-terminal-v315'

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION='track2-v317-batched-sparse-failure-terminal-v315'
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=10 TRACK2_SAVE_INTERVAL=3 TRACK2_KEEP_LAST_CHECKPOINTS=2
export TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1471 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=2e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200 TRACK2_ROLLOUT_EPOCH=4
export TRACK2_ACTOR_GLOBAL_BATCH_SIZE=3200
export TRACK2_REFERENCE_STATE_STORAGE=disk TRACK2_REFERENCE_STATE_PATH_OVERRIDE="$REFERENCE"
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true
export TRACK2_CATCH_SYSTEM_FAILURE=0 TRACK2_ACTOR_OFFLOAD=true TRACK2_ENABLE_SFT_CO_TRAIN=false
export TOKENIZERS_PARALLELISM=false TRACK2_CPUSET='6-21' TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TRACK2_RAY_SYSTEM_CONFIG_JSON='{"grpc_client_keepalive_timeout_ms":1800000,"grpc_keepalive_timeout_ms":1800000,"health_check_timeout_ms":1800000,"health_check_failure_threshold":10}'
export TRACK2_ALLOW_EXISTING_RUN=true

start_step=3
ckpt="$RUN/$EXP/checkpoints/global_step_3/actor/model_state_dict/full_weights.pt"
optimizer="$RUN/$EXP/checkpoints/global_step_3/actor/optimizer_recovery.pt"
completed=false
for attempt in $(seq 1 12); do
  "$PY" -c 'import sys,zipfile; raise SystemExit(0 if all(zipfile.is_zipfile(p) for p in sys.argv[1:]) else 1)' "$ckpt" "$optimizer"
  export TRACK2_START_STEP="$start_step" TRACK2_CKPT_PATH="$ckpt" TRACK2_OPTIMIZER_RECOVERY_PATH="$optimizer"
  printf 'attempt=%s start_utc=%s start_step=%s\n' "$attempt" "$(date -u +%FT%TZ)" "$start_step" >>"$REG/step3_resume_attempts.log"
  set +e
  bash "$P/run_strict_track2_conservative_kl.sh" >"$REG/step3_resume_attempt_${attempt}.screen.log" 2>&1
  rc=$?
  set -e
  cp -f "$RUN/launcher.log" "$REG/step3_resume_launcher_attempt_${attempt}.log" 2>/dev/null || true
  printf 'attempt=%s exit_rc=%s end_utc=%s\n' "$attempt" "$rc" "$(date -u +%FT%TZ)" >>"$REG/step3_resume_attempts.log"
  if [[ -s "$CHECKPOINT" ]] && "$PY" -c 'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$CHECKPOINT"; then
    completed=true
    break
  fi
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  readarray -t fields < <("$PY" - "$RUN" "$EXP" <<'PY'
import sys,zipfile
from pathlib import Path
run,exp=Path(sys.argv[1]),sys.argv[2]; valid=[]
for path in (run/exp/'checkpoints').glob('global_step_*'):
    try: step=int(path.name.rsplit('_',1)[1])
    except ValueError: continue
    model=path/'actor/model_state_dict/full_weights.pt'; opt=path/'actor/optimizer_recovery.pt'
    if step<10 and model.is_file() and opt.is_file() and zipfile.is_zipfile(model) and zipfile.is_zipfile(opt): valid.append((step,model,opt))
if not valid: raise SystemExit(2)
step,model,opt=max(valid,key=lambda row:row[0]); print(step); print(model); print(opt)
PY
  )
  start_step="${fields[0]}"; ckpt="${fields[1]}"; optimizer="${fields[2]}"
done

if [[ "$completed" != true ]]; then echo V318_STEP3_RESUME_EXHAUSTED >&2; exit 9; fi
"$PY" "$P/audit_strict_track2_kl_smoke.py" --run "$RUN" --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/p3_training_acceptance.json" >"$RUN/audit/p3_training_acceptance.log" 2>&1
touch "$RUN/audit/V318_TRAINING_ACCEPTED"
echo V318_RESUMED_OFFICIAL_RL_TRAINING_ACCEPTED
