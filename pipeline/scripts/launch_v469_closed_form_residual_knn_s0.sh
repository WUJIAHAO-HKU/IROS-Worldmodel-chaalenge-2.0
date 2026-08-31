#!/usr/bin/env bash
# Single-shot public-train v469 C/E KNN4 OOF only; no reward, policy update, RL, dev/final, or submission.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v469_closed_form_residual_knn_seed1616_20260824";RESULT="$REG/result";PREREG="$REG/preregistration.json"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';CONTRACT="$ROOT/pipeline/scripts/v469_closed_form_residual_knn_contract.json"
PREP="$ROOT/pipeline/scripts/prepare_v469_closed_form_residual_knn.py";PROBE="$ROOT/pipeline/scripts/probe_v469_closed_form_residual_knn_s0.py";AUDIT="$ROOT/pipeline/scripts/audit_v469_closed_form_residual_knn_s0.py"
V468="$J/v468_endpoint_residual_seed1616_20260823/preregistration.json";V461="$J/v461_endpoint200_seed1612_20260823";SELECTION="$V461/selection.json";GENERATION="$V461/dataset/generation_report.json";DATASET="$V461/dataset"
LOCK='/var/lock/v469_closed_form_residual_knn_seed1616.lock';FAILURE="$REG/launcher_failure_receipt.json";START=$(date +%s);ACTIVE='';WATCHDOG='';STOPPED=0;SUCCESS=0;REASON=launcher_exception
exec 9>"$LOCK";flock -n 9 || { echo V469_ALREADY_RUNNING;exit 73; }
test ! -e "$REG";mkdir -p "$REG"
test "$(sha256sum "$CONTRACT"|awk '{print $1}')" = 'e79b8009b07281d2d9cd2a9b4b91a127a4ba366db1a05e38ee1bce00abd8f2e8'
test "$(sha256sum "$PREP"|awk '{print $1}')" = '076caa0da57e96629c4c6d7321afcc7d9babfa66f17d90b001efd3888be69505'
test "$(sha256sum "$PROBE"|awk '{print $1}')" = '9497cdfcaae0e8a79c65fe7cb602e8a66e63790764b5e84e02056531a12ba123'
test "$(sha256sum "$AUDIT"|awk '{print $1}')" = 'b9d0e133d1b38cd804caeaa79d4ce55596d7becf588dc34c05ba7fe1077fa60b'
"$RLPY" -m py_compile "$PREP" "$PROBE" "$AUDIT"
kill_active(){ if [[ -n "$ACTIVE" ]]&&kill -0 "$ACTIVE" 2>/dev/null;then kill -TERM -- "-$ACTIVE" 2>/dev/null||true;for _ in $(seq 1 15);do kill -0 "$ACTIVE" 2>/dev/null||break;sleep 1;done;kill -KILL -- "-$ACTIVE" 2>/dev/null||true;wait "$ACTIVE" 2>/dev/null||true;fi;ACTIVE=''; }
run(){ local seconds=$1 log=$2 code;shift 2;setsid "$@" >"$log" 2>&1 & ACTIVE=$!;local start=$(date +%s);while kill -0 "$ACTIVE" 2>/dev/null;do (( $(date +%s)-start < seconds ))||{ kill_active;return 124;};sleep 1;done;if wait "$ACTIVE";then code=0;else code=$?;fi;ACTIVE='';return $code; }
restore(){ run 60 "$REG/restore_v218.log" bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start||return 1;for _ in $(seq 1 60);do curl -fsS http://127.0.0.1:8005/v1/health 2>/dev/null|grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'&&curl -fsS http://127.0.0.1:18084/health 2>/dev/null|grep -q '"status":"ready"'&&return 0;sleep 1;done;return 1; }
failure(){ "$RLPY" - "$FAILURE" "$REASON" <<'PY'
import json,os,sys,time
p=sys.argv[1];tmp=p+'.tmp';v={'format':'strict-track2-v469-launcher-failure-receipt-v1','passed':False,'reason':sys.argv[2],'wall_seconds':int(time.time()-int(os.environ['V469_START'])),'reward_loaded':False,'policy_updates':0,'rl_authorized':False}
with open(tmp,'w') as f:json.dump(v,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PY
}
cleanup(){ local code=$?;trap - EXIT TERM INT;if [[ -n "$WATCHDOG" ]];then kill "$WATCHDOG" 2>/dev/null||true;wait "$WATCHDOG" 2>/dev/null||true;fi;kill_active;if [[ "$STOPPED" -eq 1 ]];then restore||code=125;fi;if [[ "$SUCCESS" -ne 1 && ! -e "$FAILURE" ]];then failure||true;fi;exit $code; }
export V469_START="$START";trap cleanup EXIT;trap 'REASON=whole_chain_timeout_600s;exit 124' TERM;trap 'REASON=interrupted;exit 130' INT;(sleep 600;kill -TERM $$)&WATCHDOG=$!
run 120 "$REG/prepare.log" env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$RLPY" "$PREP" --contract "$CONTRACT" --v468-preregistration "$V468" --selection "$SELECTION" --generation-report "$GENERATION" --dataset "$DATASET" --probe "$PROBE" --auditor "$AUDIT" --launcher "$0" --output "$PREREG"
STOPPED=1;run 60 "$REG/stop_v218.log" bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done;test -z "$(ss -ltn|grep -E ':(8005|18084) '||true)"
GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits);[[ "$GPU" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]&&(( GPU<=1024 ))||{ REASON=gpu_not_idle;exit 121; }
set +e
run 540 "$REG/probe.log" env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-11 "$RLPY" "$PROBE" --preregistration "$PREREG" --contract "$CONTRACT" --selection "$SELECTION" --dataset "$DATASET" --output-dir "$RESULT" --device cuda
PROBE_CODE=$?
set -e
restore;STOPPED=0
run 60 "$REG/audit.log" env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$RLPY" "$AUDIT" --preregistration "$PREREG" --contract "$CONTRACT" --probe "$PROBE" --selection "$SELECTION" --dataset "$DATASET" --result-dir "$RESULT" --output "$REG/audit_receipt.json"
kill "$WATCHDOG" 2>/dev/null||true;wait "$WATCHDOG" 2>/dev/null||true;WATCHDOG=''
if [[ "$PROBE_CODE" -ne 0 ]];then REASON="s0_candidate_failed_exit_${PROBE_CODE}";failure;exit "$PROBE_CODE";fi
SUCCESS=1;trap - EXIT TERM INT;echo V469_CLOSED_FORM_RESIDUAL_KNN_S0_PASS
