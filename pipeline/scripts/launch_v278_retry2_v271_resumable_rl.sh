#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export TRACK2_CPUSET="${TRACK2_CPUSET:-0-2}"
export OMP_NUM_THREADS="${TRACK2_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${TRACK2_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${TRACK2_OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${TRACK2_NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${TRACK2_VECLIB_MAXIMUM_THREADS:-1}"
export RAYON_NUM_THREADS="${TRACK2_RAYON_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TRACK2_TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TRACK2_TF_NUM_INTEROP_THREADS:-1}"

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RELEASE="$JOINT/v273_v271_corrected_alpha_long_gate_seed1470_20260819"
MANIFEST="$RELEASE/release_registration.json"
MODEL_VERSION='track2-v271-endpoint-calibrated-terminal'
EXPERIMENT='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
CHECKPOINT="$RUN/$EXPERIMENT/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
REFERENCE="$OFF/immutable_reference_policy_state/pi05_official_rank0.pt"
MODEL_ONLY_STEP1="$RUN/$EXPERIMENT/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

restore() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
}
trap restore EXIT

if [[ ! -e "$REG" && ! -e "$RUN" ]]; then
  "$PY" "$P/prepare_v278_retry2_v271_resumable_rl.py" \
    >"$OFF/run_registry/$NAME.prepare.tmp.log" 2>&1
  mv "$OFF/run_registry/$NAME.prepare.tmp.log" "$REG/prepare.log"
elif [[ -s "$REG/preregistration.json" && ! -e "$RUN" ]]; then
  "$PY" - "$REG/preregistration.json" "$RUN" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["run_path"] == sys.argv[2]
assert d["selection_protocol"]["real_submission"] is False
assert d["guards"]["hidden_or_final_data_used_for_training_or_selection"] is False
PY
  echo REUSING_FROZEN_PRETRAINING_PREREGISTRATION >>"$REG/prepare.log"
elif [[ "${TRACK2_OPERATOR_RESTART:-0}" == 1 && -s "$REG/preregistration.json" && -d "$RUN/audit" ]]; then
  test ! -e "$RUN/audit/V278_RETRY2_TRAINING_ACCEPTED"
  test -s "$REG/cpu_limit_restart_amendment.json"
  export TRACK2_ALLOW_EXISTING_RUN=true
  echo REUSING_INTERRUPTED_RETRY2_AFTER_CPU_LIMIT_AMENDMENT >>"$REG/prepare.log"
else
  echo 'refusing ambiguous existing retry2 state' >&2
  exit 3
fi

# The old retry1 watcher can never reach an acceptance marker. Stop only that
# exact detached screen before registering the new supervised run.
screen -S watch_v275_to_v276 -X quit >/dev/null 2>&1 || true

TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/restart_v271_v274_services.sh" start \
  >"$REG/restart_v271_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN" TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=10 TRACK2_SAVE_INTERVAL="${TRACK2_SAVE_INTERVAL:-3}" TRACK2_KEEP_LAST_CHECKPOINTS=2
export TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1471 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=2e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200
export TRACK2_ROLLOUT_EPOCH=4 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=3200
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_REFERENCE_STATE_PATH_OVERRIDE="$REFERENCE"
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TRACK2_ACTOR_OFFLOAD=true TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TRACK2_RAY_SYSTEM_CONFIG_JSON='{"grpc_client_keepalive_timeout_ms":1800000,"grpc_keepalive_timeout_ms":1800000,"health_check_timeout_ms":1800000,"health_check_failure_threshold":10}'

if [[ "${TRACK2_MODEL_ONLY_RECOVERY:-0}" == 1 ]]; then
  test -s "$REG/model_only_step1_recovery_amendment.json"
  test -s "$MODEL_ONLY_STEP1"
  "$PY" -c 'import sys,zipfile; assert zipfile.is_zipfile(sys.argv[1])' "$MODEL_ONLY_STEP1"
  export TRACK2_START_STEP=1 TRACK2_CKPT_PATH="$MODEL_ONLY_STEP1"
  export TRACK2_OPTIMIZER_RECOVERY_PATH=''
fi

# On an operator restart, prefer the latest complete intermediate checkpoint
# immediately. Previously this selection happened only after another failed
# attempt, which could incorrectly fall back to the model-only step-1 state.
readarray -t prelaunch_recovery < <("$PY" - "$RUN" "$EXPERIMENT" <<'PY'
import sys, zipfile
from pathlib import Path
run, experiment = Path(sys.argv[1]), sys.argv[2]
valid = []
for path in (run / experiment / "checkpoints").glob("global_step_*"):
    try:
        step = int(path.name.rsplit("_", 1)[1])
    except ValueError:
        continue
    model = path / "actor/model_state_dict/full_weights.pt"
    optimizer = path / "actor/optimizer_recovery.pt"
    if step < 10 and model.is_file() and optimizer.is_file():
        if zipfile.is_zipfile(model) and zipfile.is_zipfile(optimizer):
            valid.append((step, model, optimizer))
if valid:
    step, model, optimizer = max(valid, key=lambda item: item[0])
    print(step); print(model); print(optimizer)
PY
)
if [[ -n "${prelaunch_recovery[0]:-}" ]]; then
  export TRACK2_START_STEP="${prelaunch_recovery[0]}"
  export TRACK2_CKPT_PATH="${prelaunch_recovery[1]}"
  export TRACK2_OPTIMIZER_RECOVERY_PATH="${prelaunch_recovery[2]}"
fi

completed=false
max_recovery_attempts="${TRACK2_MAX_RECOVERY_ATTEMPTS:-24}"
attempt_base=$(grep -c '^attempt=.* start_utc=' "$REG/recovery_attempts.log" 2>/dev/null || true)
if (( attempt_base >= max_recovery_attempts )); then
  echo 'V278_RETRY2_EXHAUSTED_RECOVERY_ATTEMPTS' >&2
  exit 9
fi
for attempt in $(seq $((attempt_base + 1)) "$max_recovery_attempts"); do
  printf 'attempt=%s start_utc=%s start_step=%s\n' \
    "$attempt" "$(date -u +%FT%TZ)" "${TRACK2_START_STEP:-0}" \
    >>"$REG/recovery_attempts.log"
  set +e
  bash "$P/run_strict_track2_conservative_kl.sh" \
    >"$REG/runner_attempt_${attempt}.screen.log" 2>&1
  rc=$?
  set -e
  cp -f "$RUN/launcher.log" "$REG/launcher_attempt_${attempt}.log" 2>/dev/null || true
  printf 'attempt=%s exit_rc=%s end_utc=%s\n' \
    "$attempt" "$rc" "$(date -u +%FT%TZ)" >>"$REG/recovery_attempts.log"
  if (( rc == 0 )); then
    completed=true
    break
  fi

  # The save-cycle watcher may deliberately release the idle CPU-offloaded
  # rollout worker to make room for the terminal full-state materialization.
  # Ray can then return non-zero after the already-flushed terminal metrics and
  # model checkpoint are complete. Treat only a valid terminal ZIP as success;
  # intermediate checkpoints still require the exact model+optimizer recovery
  # path below.
  if [[ -s "$CHECKPOINT" ]] && \
     "$PY" -c 'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$CHECKPOINT"; then
    printf 'attempt=%s terminal_checkpoint_recovered=true path=%s utc=%s\n' \
      "$attempt" "$CHECKPOINT" "$(date -u +%FT%TZ)" >>"$REG/recovery_attempts.log"
    completed=true
    break
  fi

  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  recovery_json="$REG/recovery_after_attempt_${attempt}.json"
  "$PY" - "$RUN" "$EXPERIMENT" "$recovery_json" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

run, experiment, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
root = run / experiment / "checkpoints"
valid = []
invalid = []
for path in sorted(root.glob("global_step_*")) if root.exists() else []:
    try:
        step = int(path.name.rsplit("_", 1)[1])
    except ValueError:
        invalid.append(str(path))
        continue
    model = path / "actor/model_state_dict/full_weights.pt"
    optimizer = path / "actor/optimizer_recovery.pt"
    if step < 10 and model.is_file() and optimizer.is_file() and zipfile.is_zipfile(model) and zipfile.is_zipfile(optimizer):
        valid.append((step, model, optimizer, path))
    else:
        invalid.append(str(path))
payload = {"valid_steps": [x[0] for x in valid], "invalid": invalid}
if valid:
    step, model, optimizer, path = valid[-1]
    payload["selected"] = {"step": step, "model": str(model), "optimizer": str(optimizer), "checkpoint": str(path)}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload.get("selected", {})))
PY
  selected=$(tail -n 1 "$recovery_json" 2>/dev/null || true)
  readarray -t recovery_fields < <("$PY" - "$recovery_json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])).get("selected", {})
print(d.get("step", "")); print(d.get("model", "")); print(d.get("optimizer", ""))
PY
  )
  step="${recovery_fields[0]:-}"
  model="${recovery_fields[1]:-}"
  optimizer="${recovery_fields[2]:-}"
  if [[ -n "$step" ]]; then
    export TRACK2_START_STEP="$step" TRACK2_CKPT_PATH="$model"
    export TRACK2_OPTIMIZER_RECOVERY_PATH="$optimizer"
  elif [[ "${TRACK2_MODEL_ONLY_RECOVERY:-0}" == 1 ]]; then
    export TRACK2_START_STEP=1 TRACK2_CKPT_PATH="$MODEL_ONLY_STEP1"
    export TRACK2_OPTIMIZER_RECOVERY_PATH=''
  else
    export TRACK2_START_STEP=0 TRACK2_CKPT_PATH='' TRACK2_OPTIMIZER_RECOVERY_PATH=''
  fi
  export TRACK2_ALLOW_EXISTING_RUN=true
done

if [[ "$completed" != true ]]; then
  echo 'V278_RETRY2_EXHAUSTED_RECOVERY_ATTEMPTS' >&2
  exit 9
fi

"$PY" "$P/audit_strict_track2_kl_smoke.py" --run "$RUN" \
  --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/p3_training_acceptance.json" \
  >"$RUN/audit/p3_training_acceptance.log" 2>&1
touch "$RUN/audit/V278_RETRY2_TRAINING_ACCEPTED"
test -s "$CHECKPOINT"
"$PY" -c 'import sys,zipfile; assert zipfile.is_zipfile(sys.argv[1])' "$CHECKPOINT"
echo V278_RETRY2_RESUMABLE_TRAINING_ACCEPTED
