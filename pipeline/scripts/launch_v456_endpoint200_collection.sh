#!/usr/bin/env bash
# Public-train endpoint collection only: 10 atomic batches x 20 contexts. No model/reward/policy/RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v461_endpoint200_seed1612_20260823"
V455="$J/v455_postclose_endpoint_pilot_seed1608_20260823"
V457="$J/v457_v455_immutable_reconciliation_seed1610_20260823"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
DATASET="$ROOT/artifacts/datasets/aloha-agilex_clean_50"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
PREP="$ROOT/pipeline/scripts/prepare_v456_endpoint200_action_selection.py"
GEN="$ROOT/pipeline/scripts/generate_v456_endpoint200_sharded.py"
AUDIT="$ROOT/pipeline/scripts/audit_v456_endpoint200_batches.py"
CONTRACT="$ROOT/pipeline/scripts/v456_endpoint_residual_parent_frozen_contract.json"
V460CONTRACT="$ROOT/pipeline/scripts/v460_endpoint_only_collection_contract.json"
V455COLLECTOR="$ROOT/pipeline/scripts/generate_v455_postclose_endpoint_pilot.py"
ENDPOINTCOLLECTOR="$ROOT/pipeline/scripts/collect_v460_endpoint_only.py"
SELECTION="$REG/selection.json"
SELECTION_SHA="$REG/selection.sha256"
OUT="$REG/dataset"
AUDITS="$REG/audits"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
START_EPOCH="$(date +%s)"
LOCK_FILE='/var/lock/v461_endpoint200_seed1612.lock'
WATCHDOG_PID=''

test "$(readlink -f "$("$RLPY" -c 'import sys; print(sys.executable)')")" = "$(readlink -f "$RLPY")"
"$RLPY" -c 'import numpy,h5py,scipy,yaml,open3d,toppra,mplib,sapien'
test "$(sha256sum "$PREP" | awk '{print $1}')" = '961b977ca2720182904c3eb0c83c75e7b6b2f36f6d6d5b9c1419db44ef700395'
test "$(sha256sum "$GEN" | awk '{print $1}')" = '340d811c6222854f4a3cb9add93185011ae2b80c079b3a85fcb35c9281c4aee9'
test "$(sha256sum "$AUDIT" | awk '{print $1}')" = '1efe21bb030da76efdbc3467dc505f6cb100493695126d9dbe774fb91a6ed6aa'
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = 'd23af656e4069ba0c09f61e85ef8ce108e433f53bad28f2e2e07cb9f4f4c608b'
test "$(sha256sum "$V460CONTRACT" | awk '{print $1}')" = '925aa0a6843b268725f048cae54f68b4883c781c1d655ed33e85f6ebcbbbf923'
test "$(sha256sum "$ENDPOINTCOLLECTOR" | awk '{print $1}')" = '6c64e4a42f273f43b9e64523bf12e4dc5f24618d5673848db6c1582b5547e846'
test "$(sha256sum "$V457/receipt.json" | awk '{print $1}')" = '790b8954c659c8038e778eecfba6cebdb84f8e5a477c2956d2755e58c7cf24d4'
test "$(sha256sum "$V457/preregistration.json" | awk '{print $1}')" = 'a272626c5e9022a9808088ce5458e249e429802f9b3189ddb24f2e93283562eb'
test "$(sha256sum "$V455COLLECTOR" | awk '{print $1}')" = '24f53c12adc62c56f09bba3fca3eebb65ee5020db1eec2b488fd7ac7a9b039f3'
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
exec 9>>"$LOCK_FILE"
flock -n 9
printf '{"acquired":true,"pid":%s,"run_id":"%s"}\n' "$$" "$RUN_ID" >"$LOCK_FILE"
mkdir -p "$REG" "$AUDITS"
if [[ ! -e "$SELECTION" && ! -e "$SELECTION_SHA" ]]; then
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-7 "$RLPY" "$PREP" \
    --contract "$CONTRACT" --reconciliation "$V457/receipt.json" --reconciliation-preregistration "$V457/preregistration.json" \
    --pilot-dataset "$V455/dataset" --v455-collector "$V455COLLECTOR" --split "$SPLIT" --dataset "$DATASET" \
    --seed-file "$DATASET/seed.txt" --output "$SELECTION" >"$REG/prepare.log" 2>&1
  "$RLPY" - "$SELECTION" "$SELECTION_SHA" <<'PY'
import hashlib,os,sys
source,target=sys.argv[1:];value=hashlib.sha256(open(source,'rb').read()).hexdigest();tmp=target+'.tmp'
with open(tmp,'w') as f:f.write(value+'\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,target)
PY
elif [[ -e "$SELECTION" && -e "$SELECTION_SHA" ]]; then
  :
else
  echo 'V456_SELECTION_SHA_PAIR_INCOMPLETE_MANUAL_FORENSIC_REQUIRED' >&2
  exit 2
fi
test "$(cat "$SELECTION_SHA")" = "$(sha256sum "$SELECTION" | awk '{print $1}')"

stop_watchdog() { if [[ -n "$WATCHDOG_PID" ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; wait "$WATCHDOG_PID" 2>/dev/null || true; WATCHDOG_PID=''; fi; }
restore_v218() {
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >>"$REG/restore_v218.log" 2>&1 || return 1
  for _ in $(seq 1 60); do
    if curl -fsS 'http://127.0.0.1:8005/v1/health' 2>/dev/null | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware' && curl -fsS 'http://127.0.0.1:18084/health' 2>/dev/null | grep -q '"status":"ready"'; then return 0; fi
    sleep 1
  done
  return 1
}
cleanup() { stop_watchdog; restore_v218 || echo V456_RESTORE_V218_FAILED >&2; }
trap cleanup EXIT
trap 'exit 124' TERM INT
remaining=$(( 10800 - ($(date +%s) - START_EPOCH) )); test "$remaining" -gt 0
( sleep "$remaining"; kill -TERM $$ ) & WATCHDOG_PID=$!
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

audit_common=(--selection "$SELECTION" --selection-sha-file "$SELECTION_SHA" --contract "$CONTRACT" --v460-contract "$V460CONTRACT" --reconciliation "$V457/receipt.json" --v457-preregistration "$V457/preregistration.json" --pilot-dataset "$V455/dataset" --prepare "$PREP" --generator "$GEN" --v455-collector "$V455COLLECTOR" --endpoint-collector "$ENDPOINTCOLLECTOR" --split "$SPLIT" --seed-file "$DATASET/seed.txt" --dataset "$DATASET" --output-dir "$OUT" --audit-dir "$AUDITS" --lock-file "$LOCK_FILE")
gen_common=(--reconciliation "$V457/receipt.json" --reconciliation-preregistration "$V457/preregistration.json" --pilot-dataset "$V455/dataset" --selection "$SELECTION" --contract "$CONTRACT" --v460-contract "$V460CONTRACT" --v455-collector "$V455COLLECTOR" --endpoint-collector "$ENDPOINTCOLLECTOR" --split "$SPLIT" --seed-file "$DATASET/seed.txt" --dataset "$DATASET" --support-root "$SUPPORT" --task-config "$SUPPORT/task_config/demo_clean.yml" --output-dir "$OUT")

cd "$SUPPORT"
for batch_id in $(seq 0 9); do
  elapsed=$(( $(date +%s) - START_EPOCH )); test "$elapsed" -lt 10800
  pad="$(printf '%03d' "$batch_id")"
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-7 "$RLPY" "$AUDIT" --mode pre --batch-id "$batch_id" "${audit_common[@]}" --receipt-output "$AUDITS/batch_${pad}_pre_${RUN_ID}.json" >"$AUDITS/batch_${pad}_pre_${RUN_ID}.log" 2>&1
  PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" timeout --signal=TERM --kill-after=30s 930s nice -n 10 taskset -c 0-7 "$RLPY" "$GEN" --batch-id "$batch_id" "${gen_common[@]}" >"$REG/batch_${pad}_${RUN_ID}.log" 2>&1
  PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-7 "$RLPY" "$AUDIT" --mode post --batch-id "$batch_id" "${audit_common[@]}" --receipt-output "$AUDITS/batch_${pad}_post_${RUN_ID}.json" >"$AUDITS/batch_${pad}_post_${RUN_ID}.log" 2>&1
done
elapsed=$(( $(date +%s) - START_EPOCH )); test "$elapsed" -lt 10800
if [[ ! -e "$OUT/generation_report.json" ]]; then
  PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-7 "$RLPY" "$GEN" --finalize "${gen_common[@]}" >"$REG/finalize_${RUN_ID}.log" 2>&1
fi
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-7 "$RLPY" "$AUDIT" --mode final "${audit_common[@]}" --receipt-output "$AUDITS/final_${RUN_ID}.json" >"$AUDITS/final_${RUN_ID}.log" 2>&1
elapsed=$(( $(date +%s) - START_EPOCH )); test "$elapsed" -le 10800
"$RLPY" - "$REG/launcher_receipt_${RUN_ID}.json" "$elapsed" <<'PY'
import json,sys,time
path,elapsed=sys.argv[1],int(sys.argv[2]);json.dump({"format":"strict-track2-v461-endpoint200-launcher-receipt-v1","passed":elapsed<=10800,"wall_seconds":elapsed,"exclusive_lock_acquired":True,"model_training":False,"policy_updates":0,"rl_authorized":False},open(path,"w"),indent=2)
open(path,"a").write("\n")
PY
stop_watchdog
restore_v218
trap - EXIT TERM INT
echo V456_ENDPOINT200_COLLECTION_COMPLETE
