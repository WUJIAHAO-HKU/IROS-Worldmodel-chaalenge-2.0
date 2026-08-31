#!/usr/bin/env bash
# Preregistered public-train v463 WM parent gate only. No reward, policy update, RL, dev/final, or submission.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
V461="$J/v461_endpoint200_seed1612_20260823"
REG="$J/v463_endpoint_residual_seed1614_20260823"
TRAINING="$REG/training"
RELEASE="$J/v463_v169_endpoint_residual_parent_release"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
PREP="$ROOT/pipeline/scripts/prepare_v463_endpoint_residual_5fold.py"
TRAINER="$ROOT/pipeline/scripts/train_v463_endpoint_residual_5fold.py"
PACKAGER="$ROOT/pipeline/scripts/package_v463_endpoint_residual_release.py"
RUNTIME="$ROOT/pipeline/wam_pipeline/v463_v169_endpoint_residual_runtime.py"
V169_RUNTIME="$ROOT/pipeline/wam_pipeline/v169_arm_routed_runtime.py"
V169_RELEASE="$J/v169_instruction_arm_routed_release"
V169_LIBRARY="$ROOT/artifacts"
SELECTION="$V461/selection.json"
GENERATION="$V461/dataset/generation_report.json"
DATASET="$V461/dataset"
PREREG="$REG/preregistration.json"
LOCK_FILE='/var/lock/v463_endpoint_residual_seed1614.lock'
FAILURE="$REG/launcher_failure_receipt.json"
START_EPOCH="$(date +%s)"
ACTIVE_PID=''
LAST_ACTIVE_PID=''
WATCHDOG_PID=''
STOPPED=0
SUCCESS=0
FAIL_REASON='launcher_exception'
GPU_PEAK=0

test "$(readlink -f "$("$RLPY" -c 'import sys; print(sys.executable)')")" = "$(readlink -f "$RLPY")"
"$RLPY" -c 'import numpy,h5py,torch,PIL'
test "$(sha256sum "$PREP" | awk '{print $1}')" = '9f4fbd1ee1b17eed4df237bf61e668ed6d27d80f402914e88aefd16ef23423a8'
test "$(sha256sum "$TRAINER" | awk '{print $1}')" = 'c7541a49b00d765d9dfec4b9512fd64ca2f64d4d5fb7702426bbe2ce061ba9ae'
test "$(sha256sum "$PACKAGER" | awk '{print $1}')" = '6a18453b4fad41643fcce7783d972c51e78fe8d5ce8ca198518aa17461437356'
test "$(sha256sum "$RUNTIME" | awk '{print $1}')" = '2cd396b525d64f8de2aa6aa084a425e0e0db03c9df129b50f0fd42f61b4b2ec4'
test -f "$SELECTION" -a -f "$GENERATION"
mapfile -t FINAL_RECEIPTS < <(find "$V461/audits" -maxdepth 1 -type f -name 'final_*.json' -print | sort)
test "${#FINAL_RECEIPTS[@]}" -eq 1
FINAL_RECEIPT="${FINAL_RECEIPTS[0]}"
"$RLPY" - "$FINAL_RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r.get('format')=='strict-track2-v461-endpoint200-batch-audit-v1'
assert r.get('mode')=='final' and r.get('passed') is True and all(r.get('checks',{}).values())
assert r.get('guards',{}).get('rl_authorized') is False
PY
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
exec 9>>"$LOCK_FILE"
flock -n 9
test ! -e "$REG" -a ! -e "$RELEASE"
mkdir -p "$REG"

atomic_failure() {
  local reason="$1" exit_code="$2" elapsed
  elapsed=$(( $(date +%s) - START_EPOCH ))
  "$RLPY" - "$FAILURE" "$reason" "$exit_code" "$elapsed" "$GPU_PEAK" "$LAST_ACTIVE_PID" <<'PY'
import json,os,sys
path,reason,code,elapsed,peak,pid=sys.argv[1:]
payload={'format':'strict-track2-v463-launcher-failure-receipt-v1','passed':False,'reason':reason,'exit_code':int(code),'wall_seconds':int(elapsed),'gpu_peak_mib':int(peak),'training_process_group':int(pid) if pid else None,'reward_loaded':False,'policy_updates':0,'rl_authorized':False}
tmp=path+'.tmp'
with open(tmp,'x') as f: json.dump(payload,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path); d=os.open(os.path.dirname(path),os.O_RDONLY); os.fsync(d); os.close(d)
PY
}
kill_active_group() {
  if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$ACTIVE_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  ACTIVE_PID=''
}
run_bounded() {
  local seconds="$1" log="$2" started status
  shift 2
  started="$(date +%s)"
  setsid "$@" >"$log" 2>&1 &
  ACTIVE_PID=$!
  LAST_ACTIVE_PID=$ACTIVE_PID
  while kill -0 "$ACTIVE_PID" 2>/dev/null; do
    if (( $(date +%s) - started >= seconds )); then kill_active_group; return 124; fi
    sleep 1
  done
  set +e; wait "$ACTIVE_PID"; status=$?; set -e
  ACTIVE_PID=''
  return "$status"
}
gpu_used_mib() {
  local raw
  raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  [[ "$raw" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 1
  printf '%s\n' "$raw" | tr -d '[:space:]'
}
restore_v218() {
  run_bounded 60 "$REG/restore_v218.log" bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start || return 1
  for _ in $(seq 1 60); do
    if curl -fsS 'http://127.0.0.1:8005/v1/health' 2>/dev/null | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware' && curl -fsS 'http://127.0.0.1:18084/health' 2>/dev/null | grep -q '"status":"ready"'; then return 0; fi
    sleep 1
  done
  return 1
}
cleanup() {
  local code=$?
  trap - EXIT TERM INT
  if [[ -n "$WATCHDOG_PID" ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; wait "$WATCHDOG_PID" 2>/dev/null || true; WATCHDOG_PID=''; fi
  kill_active_group
  if [[ "$STOPPED" -eq 1 ]]; then restore_v218 || code=125; fi
  if [[ "$SUCCESS" -ne 1 && ! -e "$FAILURE" ]]; then atomic_failure "$FAIL_REASON" "$code" || true; fi
  exit "$code"
}
trap cleanup EXIT
trap 'FAIL_REASON=whole_chain_hard_timeout_720s; exit 124' TERM
trap 'FAIL_REASON=launcher_interrupted; exit 130' INT
( sleep 720; kill -TERM $$ ) &
WATCHDOG_PID=$!

run_bounded 700 "$REG/prepare.log" env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$RLPY" "$PREP" \
  --v456-final-receipt "$FINAL_RECEIPT" --v456-selection "$SELECTION" --v456-generation-report "$GENERATION" --v456-dataset "$DATASET" \
  --v169-release "$V169_RELEASE" --v169-library "$V169_LIBRARY" --v169-runtime-source "$V169_RUNTIME" \
  --trainer "$TRAINER" --runtime "$RUNTIME" --output "$PREREG"
test $(( $(date +%s) - START_EPOCH )) -lt 720

STOPPED=1
run_bounded 60 "$REG/stop_v218.log" bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
idle_gpu="$(gpu_used_mib)" || { FAIL_REASON='gpu_idle_query_failed'; exit 121; }
(( idle_gpu <= 1024 )) || { FAIL_REASON="gpu_not_idle_${idle_gpu}_mib"; exit 121; }

cd "$ROOT"
setsid taskset -c 0-11 "$RLPY" "$TRAINER" \
  --preregistration "$PREREG" --v456-selection "$SELECTION" --v456-generation-report "$GENERATION" --v456-dataset "$DATASET" \
  --v169-release "$V169_RELEASE" --v169-library "$V169_LIBRARY" --output-dir "$TRAINING" --device cuda >"$REG/training.log" 2>&1 &
ACTIVE_PID=$!
LAST_ACTIVE_PID=$ACTIVE_PID
TRAIN_START="$(date +%s)"
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  elapsed=$(( $(date +%s) - START_EPOCH ))
  train_elapsed=$(( $(date +%s) - TRAIN_START ))
  current="$(gpu_used_mib)" || { FAIL_REASON='gpu_monitor_query_failed'; kill_active_group; exit 121; }
  (( current > GPU_PEAK )) && GPU_PEAK=$current
  if (( GPU_PEAK > 28672 )); then FAIL_REASON='gpu_peak_exceeded_28672_mib'; kill_active_group; exit 122; fi
  if (( train_elapsed >= 480 )) && [[ ! -f "$TRAINING.partial/s0_report.json" ]]; then FAIL_REASON='s0_hard_timeout_480s'; kill_active_group; exit 124; fi
  if (( elapsed >= 720 )); then FAIL_REASON='whole_chain_hard_timeout_720s'; kill_active_group; exit 124; fi
  sleep 2
done
set +e
wait "$ACTIVE_PID"
TRAIN_EXIT=$?
set -e
ACTIVE_PID=''
test "$TRAIN_EXIT" -eq 0 || { FAIL_REASON="trainer_exit_${TRAIN_EXIT}"; exit "$TRAIN_EXIT"; }
"$RLPY" - "$TRAINING/s0_report.json" "$TRAINING/training_report.json" <<'PY'
import json,sys
s0,final=map(lambda p:json.load(open(p)),sys.argv[1:])
assert s0.get('passed') is True
assert final.get('passed') is True and final.get('all200_training_performed') is True
PY
test $(( $(date +%s) - START_EPOCH )) -lt 720
remaining=$(( 720 - ($(date +%s) - START_EPOCH) )); (( remaining > 0 )) || { FAIL_REASON='whole_chain_hard_timeout_720s'; exit 124; }
run_bounded "$remaining" "$REG/package.log" env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$RLPY" "$PACKAGER" \
  --preregistration "$PREREG" --training-dir "$TRAINING" --runtime "$RUNTIME" --trainer "$TRAINER" \
  --v169-release "$V169_RELEASE" --v169-library "$V169_LIBRARY" --output "$RELEASE"
test $(( $(date +%s) - START_EPOCH )) -le 720
restore_v218
STOPPED=0
SUCCESS=1
kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
WATCHDOG_PID=''
trap - EXIT TERM INT
echo V463_ENDPOINT_RESIDUAL_S0_AND_ALL200_COMPLETE
