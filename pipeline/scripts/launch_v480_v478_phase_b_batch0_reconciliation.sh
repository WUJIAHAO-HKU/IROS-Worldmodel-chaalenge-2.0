#!/usr/bin/env bash
# CPU-only immutable batch-0 audit reconciliation. Never runs simulator/generator.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v480_v478_phase_b_batch0_reconciliation_20260824"
PRE="$REG/preregistration.json"
RECON="$S/audit_v480_v478_phase_b_batch0_reconciliation.py"
OLD_AUD="$S/audit_v478_temporal200_sharded.py"
FIXED_AUD="$S/audit_v480_v478_temporal200_sharded.py"
CONTRACT="$S/v480_v478_batch0_immutable_reconciliation_contract.json"
MATERIALIZER="$S/materialize_v480_v478_batch0_reconciliation_preregistration.py"
RECOMPUTED="$REG/recomputed_batch_000_audit.json"
OUT="$REG/reconciliation_receipt.json"
LOG="$REG/reconciliation.log"
SUCCESS="$REG/launcher_receipt.json"
FAILURE="$REG/launcher_failure_receipt.json"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

exec 9>/var/lock/v480_v478_phase_b_batch0_reconciliation.lock
flock -n 9 || exit 73
test "$(sha256sum "$PRE"|awk '{print $1}')" = '40d6116fff8d42dc279f543b7a5be9d995051ed8264e92b984ea2b9be5487536'
test "$(sha256sum "$RECON"|awk '{print $1}')" = '5afc67521bc35f2de8636a972041319b47874f80565f029d75f9577c9a2edd06'
test "$(sha256sum "$OLD_AUD"|awk '{print $1}')" = '2e1adbe72e2d77d19a2bf6f4a3d0822940ca458055851c36e06b28fe62c6c748'
test "$(sha256sum "$FIXED_AUD"|awk '{print $1}')" = '0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0'
test "$(sha256sum "$CONTRACT"|awk '{print $1}')" = '2a492ddd0b974a74c5cbd8f1d11b093905c74f931da76abd745d94fea6aec827'
test "$(sha256sum "$MATERIALIZER"|awk '{print $1}')" = '501ed98a64a1707e5063eaa508074ca8494688f1760f129f81851f044b626e4c'
for path in "$RECOMPUTED" "$RECOMPUTED.tmp" "$OUT" "$OUT.tmp" "$LOG" "$SUCCESS" "$FAILURE";do test ! -e "$path";done
curl -fsS http://127.0.0.1:8005/v1/health >/dev/null
curl -fsS http://127.0.0.1:18084/health >/dev/null
PYTHONPYCACHEPREFIX=/dev/shm/v480_pycache "$RLPY" -m py_compile "$RECON" "$FIXED_AUD"

ACTIVE_PID='';SUCCESSFUL=0;STAGE='reconciliation'
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]]&&kill -0 "$ACTIVE_PID" 2>/dev/null;then
  kill -TERM -- "-$ACTIVE_PID" 2>/dev/null||true
  for _ in $(seq 1 10);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done
  kill -KILL -- "-$ACTIVE_PID" 2>/dev/null||true;wait "$ACTIVE_PID" 2>/dev/null||true
 fi
 ACTIVE_PID=''
}
write_failure(){
 local rc="$1";[[ -e "$FAILURE" ]]&&return
 V480_STAGE="$STAGE" V480_RC="$rc" "$RLPY" - "$FAILURE" "$PRE" "$RECON" "$FIXED_AUD" "$LOG" "$OUT" "$RECOMPUTED" "$0" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ev(p):p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
out=Path(sys.argv[1]);tmp=out.with_name(out.name+'.tmp')
obj={'format':'strict-track2-v480-v478-phase-b-batch0-reconciliation-launcher-failure-v1','passed':False,'stage':os.environ['V480_STAGE'],'exit_code':int(os.environ['V480_RC']),'preregistration_sha256':sha(sys.argv[2]),'reconciler_sha256':sha(sys.argv[3]),'fixed_auditor_sha256':sha(sys.argv[4]),'log':ev(sys.argv[5]),'reconciliation_receipt':ev(sys.argv[6]),'recomputed_batch0_audit':ev(sys.argv[7]),'launcher_path':str(Path(sys.argv[8]).resolve()),'launcher_sha256':sha(sys.argv[8]),'simulator_or_collector_invocations':0,'batch0_generator_invocations':0,'phase_b_resume_authorized':False,'batch0_rerun_authorized':False,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
cleanup(){ local rc=$?;terminate_active;if [[ -f "$SUCCESS" ]];then SUCCESSFUL=1;fi;if ((SUCCESSFUL==0));then write_failure "$rc"||true;fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$RLPY" "$RECON" --preregistration "$PRE" --recomputed-audit "$RECOMPUTED" --output "$OUT" >"$LOG" 2>&1 & ACTIVE_PID=$!
deadline=$((SECONDS+300))
while kill -0 "$ACTIVE_PID" 2>/dev/null;do ((SECONDS<deadline))||{ STAGE='timeout';terminate_active;exit 124;};sleep 1;done
wait "$ACTIVE_PID";ACTIVE_PID=''

STAGE='result_validation'
"$RLPY" - "$PRE" "$RECON" "$FIXED_AUD" "$RECOMPUTED" "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));audit=json.load(open(sys.argv[4]));result=json.load(open(sys.argv[5]))
assert sha(sys.argv[1])=='40d6116fff8d42dc279f543b7a5be9d995051ed8264e92b984ea2b9be5487536' and sha(sys.argv[2])=='5afc67521bc35f2de8636a972041319b47874f80565f029d75f9577c9a2edd06' and sha(sys.argv[3])=='0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0'
assert result['format']=='strict-track2-v480-v478-phase-b-batch0-immutable-reconciliation-v1' and result['passed'] is True and all(result['checks'].values())
assert audit['format']=='strict-track2-v478-public-train-temporal200-batch-audit-v1' and audit['passed'] is True and len(audit['checks'])==7 and all(audit['checks'].values())
assert result['recomputed_batch0_audit']['sha256']==sha(sys.argv[4])
assert result['authorization']=={'phase_b_resume_authorized':True,'next_batch_id':1,'batch0_reuse_required':True,'batch0_rerun_authorized':False,'effect_or_outcome_conditioned_retry':False,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
PY
curl -fsS http://127.0.0.1:8005/v1/health >/dev/null
curl -fsS http://127.0.0.1:18084/health >/dev/null

STAGE='success_receipt'
"$RLPY" - "$SUCCESS" "$PRE" "$RECON" "$FIXED_AUD" "$RECOMPUTED" "$OUT" "$LOG" "$0" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
out=Path(sys.argv[1]);tmp=out.with_name(out.name+'.tmp')
obj={'format':'strict-track2-v480-v478-phase-b-batch0-reconciliation-launcher-v1','passed':True,'preregistration_sha256':sha(sys.argv[2]),'reconciler_sha256':sha(sys.argv[3]),'fixed_auditor_sha256':sha(sys.argv[4]),'recomputed_batch0_audit_sha256':sha(sys.argv[5]),'reconciliation_receipt_sha256':sha(sys.argv[6]),'log_sha256':sha(sys.argv[7]),'launcher_path':str(Path(sys.argv[8]).resolve()),'launcher_sha256':sha(sys.argv[8]),'simulator_or_collector_invocations':0,'batch0_generator_invocations':0,'phase_b_resume_authorized':True,'next_batch_id':1,'batch0_reuse_required':True,'batch0_rerun_authorized':False,'effect_or_outcome_conditioned_retry':False,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False,'v218_health_unchanged':True}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
SUCCESSFUL=1
trap - EXIT INT TERM
exit 0
