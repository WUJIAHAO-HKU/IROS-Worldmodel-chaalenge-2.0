#!/usr/bin/env bash
# CPU-only immutable reconciliation of the v478 oracle12 audit serialization failure.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v479_v478_endpoint_oracle12_reconciliation_20260824"
PRE="$REG/preregistration_r3.json"
RECON="$S/audit_v479_v478_oracle12_reconciliation.py"
OLD_AUD="$S/audit_v478_endpoint_oracle12.py"
ORACLE='/root/v478_endpoint_oracle12_seed1622_20260824'
LEGACY_OUT="$REG/recomputed_v478_audit_receipt.json"
OUT="$REG/reconciliation_receipt.json"
LOG="$REG/reconciliation.log"
SUCCESS="$REG/launcher_receipt.json"
FAILURE="$REG/launcher_failure_receipt.json"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

exec 9>/var/lock/v479_v478_oracle12_reconciliation.lock
flock -n 9 || exit 73
test -s "$PRE" && test -s "$RECON" && test -s "$OLD_AUD"
test "$(sha256sum "$PRE"|awk '{print $1}')" = '07b6d5974c5891068160ec9ffc69a14717cb110de0009393ad5f051473cb9ae1'
test "$(sha256sum "$RECON"|awk '{print $1}')" = '5b5a8839b7166bcbc0cf7d337300bac04c9d92ae5c90a68eed49b910e9f6a096'
test "$(sha256sum "$OLD_AUD"|awk '{print $1}')" = '6cb920fbccfa039ac3a12839ee2626614a6498397adb563c3b91cd5c573cb4a2'
test ! -e "$LEGACY_OUT" && test ! -e "$LEGACY_OUT.tmp" && test ! -e "$OUT" && test ! -e "$OUT.tmp"
test ! -e "$LOG" && test ! -e "$SUCCESS" && test ! -e "$FAILURE"
curl -fsS http://127.0.0.1:8005/v1/health >/dev/null
curl -fsS http://127.0.0.1:18084/health >/dev/null
PYTHONPYCACHEPREFIX=/dev/shm/v479_pycache "$RLPY" -m py_compile "$RECON"

ACTIVE_PID=''; SUCCESSFUL=0; STAGE='reconciliation'
terminate_active(){
  if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$ACTIVE_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$ACTIVE_PID" 2>/dev/null || true
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  ACTIVE_PID=''
}
write_failure(){
  local rc="$1"
  [[ -e "$FAILURE" ]] && return
  "$RLPY" - "$FAILURE" "$PRE" "$RECON" "$LOG" "$OUT" "$LEGACY_OUT" "$STAGE" "$rc" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ev(p):p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
out=Path(sys.argv[1]);tmp=out.with_name(out.name+'.tmp')
obj={'format':'strict-track2-v479-v478-endpoint-oracle12-reconciliation-launcher-failure-v1','passed':False,'stage':sys.argv[7],'exit_code':int(sys.argv[8]),'preregistration_sha256':sha(sys.argv[2]),'reconciler_sha256':sha(sys.argv[3]),'log':ev(sys.argv[4]),'reconciliation_receipt':ev(sys.argv[5]),'recomputed_v478_audit':ev(sys.argv[6]),'endpoint_oracle_retry_authorized':False,'simulator_execution_authorized':False,'phase_b_temporal_collection_authorized':False,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
cleanup(){
  local rc=$?
  terminate_active
  if ((SUCCESSFUL==0)); then write_failure "$rc" || true; fi
}
trap cleanup EXIT INT TERM

setsid env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$RLPY" "$RECON" --preregistration "$PRE" --legacy-output "$LEGACY_OUT" --output "$OUT" >"$LOG" 2>&1 &
ACTIVE_PID=$!
deadline=$((SECONDS+120))
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  if ((SECONDS>=deadline)); then STAGE='reconciliation_timeout'; terminate_active; exit 124; fi
  sleep 1
done
wait "$ACTIVE_PID"; ACTIVE_PID=''

STAGE='result_validation'
"$RLPY" - "$PRE" "$RECON" "$OLD_AUD" "$OUT" "$LEGACY_OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));result=json.load(open(sys.argv[4]));old=json.load(open(sys.argv[5]))
assert sha(sys.argv[1])=='07b6d5974c5891068160ec9ffc69a14717cb110de0009393ad5f051473cb9ae1'
assert sha(sys.argv[2])=='5b5a8839b7166bcbc0cf7d337300bac04c9d92ae5c90a68eed49b910e9f6a096'
assert sha(sys.argv[3])=='6cb920fbccfa039ac3a12839ee2626614a6498397adb563c3b91cd5c573cb4a2'
assert result['format']=='strict-track2-v479-v478-endpoint-oracle12-immutable-reconciliation-v1' and result['passed'] is True and all(result['checks'].values())
assert result['authorization']=={'phase_b_temporal_collection_authorized':True,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
assert old['format']=='strict-track2-v478-endpoint-oracle12-audit-v1' and old['passed'] is True and all(old['checks'].values())
assert result['recomputed_v478_audit']['sha256']==sha(sys.argv[5])
PY
curl -fsS http://127.0.0.1:8005/v1/health >/dev/null
curl -fsS http://127.0.0.1:18084/health >/dev/null

STAGE='success_receipt'
"$RLPY" - "$SUCCESS" "$PRE" "$RECON" "$OUT" "$LEGACY_OUT" "$LOG" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
out=Path(sys.argv[1]);tmp=out.with_name(out.name+'.tmp')
obj={'format':'strict-track2-v479-v478-endpoint-oracle12-reconciliation-launcher-v1','passed':True,'preregistration_sha256':sha(sys.argv[2]),'reconciler_sha256':sha(sys.argv[3]),'reconciliation_receipt_sha256':sha(sys.argv[4]),'recomputed_v478_audit_sha256':sha(sys.argv[5]),'log_sha256':sha(sys.argv[6]),'simulator_execution_count':0,'endpoint_oracle_retry_count':0,'phase_b_temporal_collection_authorized':True,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False,'v218_health_restored':True}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
SUCCESSFUL=1
trap - EXIT INT TERM
exit 0
