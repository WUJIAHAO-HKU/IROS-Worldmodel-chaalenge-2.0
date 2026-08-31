#!/usr/bin/env bash
# v480 resume of Phase-B temporal200 from immutable registered batch 0.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v480_v478_phase_b_resume_seed1622_20260824"
PRE="$REG/phase_b_resume_preregistration.json"
SEL="$J/v478_temporal200_seed1622_20260824/selection.json"
DATASET='/root/v478_temporal8_dataset_seed1622_20260824'
COL="$S/collect_v477_temporal_paired.py"
GEN="$S/generate_v478_temporal200_sharded.py"
AUD="$S/audit_v480_v478_temporal200_sharded.py"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
TASK="$SUPPORT/task_config/demo_clean.yml"
RESIZE="$ROOT/pipeline/wam_pipeline/data.py"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
AUDDIR="$REG/phase_b_batch_audits"
FINAL_AUDIT="$REG/phase_b_final_audit.json"
SUCCESS_RECEIPT="$REG/phase_b_launcher_receipt.json"
ATTEMPTS="$REG/phase_b_launcher_attempts"

exec 9>/var/lock/v480_v478_temporal200_phase_b_resume.lock
flock -n 9 || exit 73
START_EPOCH=$(date +%s)
LAUNCHER_PATH=$(readlink -f "$0")
mkdir -p "$AUDDIR" "$ATTEMPTS"

# Recover every prior, interrupted launcher invocation into an immutable receipt
# and charge its full elapsed time to the global 12-hour budget.  No data are
# deleted and no technical-effect result controls this recovery.
SCAN_JSON=$("$RLPY" - "$ATTEMPTS" "$LAUNCHER_PATH" "$PRE" "$GEN" "$AUD" "$COL" <<'PY'
import hashlib,json,math,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve(); launcher,pre,gen,aud,col=map(Path,sys.argv[2:7]);total=3232;successes=[];terminal_failures=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def files(d,exclude=()):
 out={}
 for p in sorted(d.rglob('*')):
  if p.is_symlink(): raise RuntimeError(f'symlink in attempt tree: {p}')
  if p.is_file() and p.name not in exclude: out[p.relative_to(d).as_posix()]=sha(p)
 return out
for d in sorted(root.iterdir()):
 if not d.is_dir() or d.is_symlink(): raise RuntimeError(f'unexpected attempt entry: {d}')
 receipts=[p for p in (d/'success_receipt.json',d/'failure_receipt.json',d/'interrupted_attempt_receipt.json') if p.is_file()]
 if len(receipts)>1: raise RuntimeError(f'multiple terminal receipts: {d}')
 if not receipts:
  intent=d/'attempt_intent.json'
  if not intent.is_file(): raise RuntimeError(f'orphan attempt without immutable intent: {d}')
  x=json.load(intent.open());start=int(x['start_epoch'])
  assert x['format']=='strict-track2-v478-temporal200-phase-b-attempt-intent-v1'
  assert x['launcher_path']==str(launcher.resolve()) and x['launcher_sha256']==sha(launcher) and x['preregistration_sha256']==sha(pre)
  wall=max(0,int(time.time())-start)
  out=d/'interrupted_attempt_receipt.json'; tmp=out.with_name(out.name+'.tmp')
  obj={'format':'strict-track2-v478-temporal200-phase-b-interrupted-attempt-v1','passed':False,'launcher_wall_seconds':wall,'launcher_path':str(launcher.resolve()),'launcher_sha256':sha(launcher),'preregistration_sha256':sha(pre),'generator_sha256':sha(gen),'auditor_sha256':sha(aud),'collector_sha256':sha(col),'attempt_files_sha256':files(d,('interrupted_attempt_receipt.json','interrupted_attempt_receipt.json.tmp')),'effect_or_outcome_conditioned_retry':False,'v218_health_restored':False,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
  with tmp.open('x') as f: json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,out); fd=os.open(str(d),os.O_RDONLY);os.fsync(fd);os.close(fd);receipts=[out]
 r=json.load(receipts[0].open()); wall=float(r['launcher_wall_seconds'])
 assert r['format'] in ('strict-track2-v478-temporal200-phase-b-launcher-attempt-v2','strict-track2-v478-temporal200-phase-b-interrupted-attempt-v1')
 assert r['effect_or_outcome_conditioned_retry'] is False
 if receipts[0].name=='success_receipt.json': assert r['passed'] is True and r['format']=='strict-track2-v478-temporal200-phase-b-launcher-attempt-v2'
 else: assert r['passed'] is False
 assert r['training_authorized'] is False and r['s1_authorized'] is False and r['zero_update_authorized'] is False and r['policy_updates']==0 and r['rl_authorized'] is False
 assert r['launcher_path']==str(launcher.resolve()) and r['launcher_sha256']==sha(launcher) and r['preregistration_sha256']==sha(pre)
 assert r['generator_sha256']==sha(gen) and r['auditor_sha256']==sha(aud) and r['collector_sha256']==sha(col)
 expected=files(d,('success_receipt.json','failure_receipt.json','interrupted_attempt_receipt.json'))
 assert r['attempt_files_sha256']==expected
 if not math.isfinite(wall) or wall<0: raise RuntimeError(f'invalid attempt wall: {d}')
 if r['format'].endswith('attempt-v2'):
  assert set(r)=={'format','passed','stage','batch_id','exit_code','launcher_wall_seconds','prior_launcher_wall_seconds','cumulative_launcher_wall_seconds','launcher_path','launcher_sha256','preregistration_sha256','generator_sha256','auditor_sha256','collector_sha256','stage_log','generation_report','final_audit','attempt_files_sha256','restore_v218_logs','v218_health','effect_or_outcome_conditioned_retry','v218_health_restored','training_authorized','s1_authorized','zero_update_authorized','policy_updates','rl_authorized'}
  prior=float(r['prior_launcher_wall_seconds']);cumulative=float(r['cumulative_launcher_wall_seconds'])
  assert math.isfinite(prior) and prior>=0 and math.isfinite(cumulative) and abs(cumulative-(prior+wall))<1e-6
  assert isinstance(r['stage'],str) and r['stage'] and isinstance(r['batch_id'],int) and isinstance(r['exit_code'],int)
  assert set(r['v218_health'])=={'8005_v1_health_http_code','18084_health_http_code'} and all(isinstance(x,str) for x in r['v218_health'].values())
  actual_restore=[{'exists':True,'sha256':sha(p)} for p in sorted(d.glob('restore_v218_*.log'))]
  assert r['restore_v218_logs']==actual_restore and (not r['v218_health_restored'] or r['v218_health']=={'8005_v1_health_http_code':'200','18084_health_http_code':'200'})
 if receipts[0].name=='success_receipt.json':
  assert r['v218_health_restored'] is True and r['v218_health']=={'8005_v1_health_http_code':'200','18084_health_http_code':'200'};successes.append(str(receipts[0].resolve()))
 if receipts[0].name=='failure_receipt.json' and str(r.get('stage','')).startswith('terminal_integrity_or_diagnostic_failure_') and r.get('v218_health_restored') is True and r.get('v218_health')=={'8005_v1_health_http_code':'200','18084_health_http_code':'200'}:terminal_failures.append(str(receipts[0].resolve()))
 total+=math.ceil(wall)
if len(successes)>1:raise RuntimeError('multiple successful attempts')
print(json.dumps({'prior_wall_seconds':total,'inner_success':successes[0] if successes else None,'terminal_failure':terminal_failures[-1] if terminal_failures else None},sort_keys=True))
PY
)
PRIOR_WALL=$("$RLPY" -c 'import json,sys;print(json.loads(sys.argv[1])["prior_wall_seconds"])' "$SCAN_JSON")
INNER_SUCCESS=$("$RLPY" -c 'import json,sys;print(json.loads(sys.argv[1])["inner_success"] or "")' "$SCAN_JSON")
TERMINAL_FAILURE=$("$RLPY" -c 'import json,sys;print(json.loads(sys.argv[1])["terminal_failure"] or "")' "$SCAN_JSON")
if [[ -n "$INNER_SUCCESS" ]];then
 if [[ -e "$SUCCESS_RECEIPT" ]];then cmp -s "$INNER_SUCCESS" "$SUCCESS_RECEIPT";else cp --reflink=never "$INNER_SUCCESS" "$SUCCESS_RECEIPT.tmp";sync -f "$SUCCESS_RECEIPT.tmp";mv "$SUCCESS_RECEIPT.tmp" "$SUCCESS_RECEIPT";sync -f "$REG";fi
 curl -fsS http://127.0.0.1:8005/v1/health >/dev/null && curl -fsS http://127.0.0.1:18084/health >/dev/null
 exit 0
fi
if [[ -n "$TERMINAL_FAILURE" ]];then
 test -s "$FINAL_AUDIT"
 if curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1 && curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1;then exit 3;fi
fi
((PRIOR_WALL<43200))
test ! -e "$SUCCESS_RECEIPT"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$"
ATTEMPT="$ATTEMPTS/$RUN_ID"
PREP="$ATTEMPTS/.attempt-prep-$RUN_ID"
"$RLPY" - "$PREP" "$ATTEMPT" "$START_EPOCH" "$LAUNCHER_PATH" "$PRE" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
prep,target=map(Path,sys.argv[1:3]);prep.mkdir()
obj={'format':'strict-track2-v478-temporal200-phase-b-attempt-intent-v1','start_epoch':int(sys.argv[3]),'planned_destination':str(target.resolve()),'launcher_path':str(Path(sys.argv[4]).resolve()),'launcher_sha256':sha(sys.argv[4]),'preregistration_sha256':sha(sys.argv[5])}
p=prep/'attempt_intent.json'
with p.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(prep),os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(prep,target);fd=os.open(str(target.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
ACTIVE_PID='';STAGE='preflight';CURRENT_BATCH=-1;SUCCESS=0;STAGE_LOG='';RESTORED=0;RESTORE_COUNT=0;RESTORE_LOG=''
INNER_SUCCESS_RECEIPT="$ATTEMPT/success_receipt.json"
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null;then
  kill -TERM -- "-$ACTIVE_PID" 2>/dev/null||true
  for _ in $(seq 1 15);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done
  kill -KILL -- "-$ACTIVE_PID" 2>/dev/null||true;wait "$ACTIVE_PID" 2>/dev/null||true
 fi
 ACTIVE_PID=''
}
restore(){
 terminate_active
 RESTORE_LOG="$ATTEMPT/restore_v218_$(printf '%02d' "$RESTORE_COUNT").log";RESTORE_COUNT=$((RESTORE_COUNT+1))
 test ! -e "$RESTORE_LOG"
 setsid bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$RESTORE_LOG" 2>&1 & ACTIVE_PID=$!
 local deadline=$((SECONDS+120))
 while kill -0 "$ACTIVE_PID" 2>/dev/null;do ((SECONDS<deadline))||{ terminate_active;return 1;};sleep 1;done
 wait "$ACTIVE_PID"||true;ACTIVE_PID=''
 for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&return 0;sleep 1;done
 return 1
}
write_attempt_receipt(){
 local path="$1" passed="$2" rc="$3"
 [[ -e "$path" ]]&&return
 local h8005 h18084
 h8005=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8005/v1/health 2>/dev/null||true)
 h18084=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:18084/health 2>/dev/null||true)
 V478_STAGE="$STAGE" V478_BATCH="$CURRENT_BATCH" V478_PASSED="$passed" V478_RC="$rc" V478_START="$START_EPOCH" V478_PRIOR="$PRIOR_WALL" V478_RESTORED="$RESTORED" V478_H8005="$h8005" V478_H18084="$h18084" "$RLPY" - "$path" "$PRE" "$GEN" "$AUD" "$COL" "$LAUNCHER_PATH" "$STAGE_LOG" "$DATASET/generation_report.json" "$FINAL_AUDIT" "$ATTEMPT" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ev(p):p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
out=Path(sys.argv[1]);tmp=out.with_name(out.name+'.tmp');passed=os.environ['V478_PASSED']=='1';attempt=Path(sys.argv[10]).resolve()
tree={}
for p in sorted(attempt.rglob('*')):
 if p.is_symlink():raise RuntimeError(f'symlink in attempt: {p}')
 if p.is_file() and p.resolve() not in (out.resolve(),tmp.resolve()):tree[p.relative_to(attempt).as_posix()]=sha(p)
wall=max(0,time.time()-int(os.environ['V478_START']));prior=int(os.environ['V478_PRIOR']);h8005=os.environ['V478_H8005'];h18084=os.environ['V478_H18084'];restored=os.environ['V478_RESTORED']=='1' and h8005=='200' and h18084=='200';restore_logs=[ev(p) for p in sorted(attempt.glob('restore_v218_*.log'))]
obj={'format':'strict-track2-v478-temporal200-phase-b-launcher-attempt-v2','passed':passed,'stage':os.environ['V478_STAGE'],'batch_id':int(os.environ['V478_BATCH']),'exit_code':int(os.environ['V478_RC']),'launcher_wall_seconds':wall,'prior_launcher_wall_seconds':prior,'cumulative_launcher_wall_seconds':prior+wall,'launcher_path':str(Path(sys.argv[6]).resolve()),'launcher_sha256':sha(sys.argv[6]),'preregistration_sha256':sha(sys.argv[2]),'generator_sha256':sha(sys.argv[3]),'auditor_sha256':sha(sys.argv[4]),'collector_sha256':sha(sys.argv[5]),'stage_log':ev(sys.argv[7]),'generation_report':ev(sys.argv[8]),'final_audit':ev(sys.argv[9]),'attempt_files_sha256':tree,'restore_v218_logs':restore_logs,'v218_health':{'8005_v1_health_http_code':h8005,'18084_health_http_code':h18084},'effect_or_outcome_conditioned_retry':False,'v218_health_restored':restored,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
if passed: assert restored and obj['cumulative_launcher_wall_seconds']<=43200
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
within_whole(){ (( PRIOR_WALL + $(date +%s) - START_EPOCH < 43200 )); }
on_exit(){ local rc=$?;terminate_active;if [[ -e "$INNER_SUCCESS_RECEIPT" ]];then SUCCESS=1;fi;if ((SUCCESS==0));then RESTORED=0;restore&&RESTORED=1||true;STAGE="${STAGE}_restore${RESTORED}";write_attempt_receipt "$ATTEMPT/failure_receipt.json" 0 "$rc"||true;fi; }
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test "$(sha256sum "$PRE"|awk '{print $1}')" = 'a1614185001ab98981bd394b76fac6b7220bd9e6878fff1098a712750c9a8f79'
test "$(sha256sum "$GEN"|awk '{print $1}')" = 'a20f5e9d702fa6febe946d9f713f4713ef035705b35119c00ca328046da70a58'
test "$(sha256sum "$AUD"|awk '{print $1}')" = '0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0'
test "$(sha256sum "$COL"|awk '{print $1}')" = 'be90f3dd1272706af6d9bc1f461543c9b399071139a4ea243bd962b4b01f187a'
test "$(sha256sum "$SEL"|awk '{print $1}')" = 'f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398'
RECOVER_FINAL=0;GENERATION_PRESENT=0
if [[ -e "$FINAL_AUDIT.tmp" ]];then
 STAGE='isolate_incomplete_final_audit'
 "$RLPY" - "$FINAL_AUDIT.tmp" "$ATTEMPT/incomplete_final_audit.json.tmp" <<'PY'
import os,sys
from pathlib import Path
s,d=map(Path,sys.argv[1:3]);assert s.is_file() and not s.is_symlink() and not d.exists();os.replace(s,d)
for p in (s.parent,d.parent):fd=os.open(str(p),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
fi
[[ -e "$DATASET/generation_report.json" ]]&&GENERATION_PRESENT=1
if [[ -e "$FINAL_AUDIT" ]];then RECOVER_FINAL=1;((GENERATION_PRESENT==1));fi
PYTHONPYCACHEPREFIX=/dev/shm/v478_phaseb_pycache "$RLPY" -m py_compile "$COL" "$GEN" "$AUD"
export PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
"$RLPY" - "$PRE" "$SEL" "$COL" "$GEN" "$AUD" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));c=pre['execution_closure']
assert pre['format']=='strict-track2-v478-public-train-temporal200-preregistration-v1' and pre['status']=='preregistered_public_train_temporal_collection_authorized'
for key,arg in zip(('selection','collector','generator','auditor'),sys.argv[2:6]):
 assert Path(c[key]['path']).resolve()==Path(arg).resolve() and c[key]['sha256']==sha(arg)
assert sha(c['phase_b_contract']['path'])==c['phase_b_contract']['sha256']=='a478b6f8e24d4a2312fcd4f62d294ad681b35ea4c19c1162f7dae7f9748047d1'
assert sha(c['endpoint_oracle_union']['path'])==c['endpoint_oracle_union']['sha256']=='0d3b1328709ba1b09bb29b1423596e5671fdc37c49215ea4b4f168b9b5fd53cc'
assert sha(c['v479_reconciliation_receipt']['path'])==c['v479_reconciliation_receipt']['sha256']=='12ce84ccf3cbf43720f0be0a1e857fcd2c160628a2ab169b006ef8704995b9e3'
assert sha(c['v480_reconciliation_preregistration']['path'])==c['v480_reconciliation_preregistration']['sha256']=='40d6116fff8d42dc279f543b7a5be9d995051ed8264e92b984ea2b9be5487536'
assert sha(c['v480_recomputed_batch0_audit']['path'])==c['v480_recomputed_batch0_audit']['sha256']=='713c6976a42e817d1f66414b30b35c6e72f505a2a4f68e9a4b778adc8bca04b9'
assert sha(c['v480_reconciliation_receipt']['path'])==c['v480_reconciliation_receipt']['sha256']=='4cc7d26eccf4b36a4fee4fe7f7f8ad23426b16fd1791cd37988745edca4eeed1'
assert sha(c['v480_reconciliation_launcher_receipt']['path'])==c['v480_reconciliation_launcher_receipt']['sha256']=='332873cba84cea5b5f5e4c7d9576a24f09292b6a2b863d985219b4cdc3965558'
assert sha(c['v480_reconciliation_contract']['path'])==c['v480_reconciliation_contract']['sha256']=='2a492ddd0b974a74c5cbd8f1d11b093905c74f931da76abd745d94fea6aec827'
assert sha(c['v480_resume_materializer']['path'])==c['v480_resume_materializer']['sha256']=='e444ce4c9cd0c7a4fef71cc7f2f3dcb5c0c325c4f3d1207ed561ee9714fa2826'
assert sha(c['parent_launcher_failure_receipt']['path'])==c['parent_launcher_failure_receipt']['sha256']=='48c95faeda8992ebf041d677a1b3e231b4619269f5e86574c3f8a73473ac5452'
assert pre['dataset_root']=='/root/v478_temporal8_dataset_seed1622_20260824' and len(pre['contexts'])==200
r=pre['resume_registration'];w=pre['wall_accounting'];g=pre['guards']
assert r['authorized'] is True and r['next_batch_id']==1 and r['batch0_reuse_required'] is True and r['batch0_rerun_authorized'] is False and r['batch0_generator_invocations_allowed']==0 and r['effect_or_outcome_conditioned_retry'] is False
assert Path(r['registered_batch0_audit']['path']).resolve()==Path(pre['execution_closure']['registered_batch0_audit']['path']).resolve() and r['registered_batch0_audit']['sha256']==sha(r['registered_batch0_audit']['path'])=='713c6976a42e817d1f66414b30b35c6e72f505a2a4f68e9a4b778adc8bca04b9'
assert w=={'accounting_difference_seconds':17.3342063133604,'dataset_and_launcher_accounting_are_distinct':True,'launcher_watchdog_prior_wall_seconds_ceil':3232,'prior_dataset_cumulative_attempt_wall_seconds':3214.017167359125,'prior_launcher_cumulative_wall_seconds':3231.3513736724854,'remaining_dataset_attempt_wall_seconds':39985.982832640875,'remaining_launcher_wall_seconds':39968.648626327515,'v480_reconciliation_excluded_from_collection_wall':True,'whole_wall_seconds_max':43200.0}
assert g['effect_used_for_retry_filter_batch_gate'] is False and g['next_batch_id']==1 and g['batch0_reuse_required'] is True and g['batch0_rerun_authorized'] is False and g['batch0_generator_invocations_allowed']==0 and g['effect_or_outcome_conditioned_retry'] is False and g['training_authorized'] is False and g['s1_authorized'] is False and g['zero_update_authorized'] is False and g['policy_updates']==0 and g['rl_authorized'] is False
PY
within_whole

USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}')
[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=1024))
for name in wm_v218_bridge wm_v218_gpu;do screen -S "$name" -X quit >/dev/null 2>&1||true;done
for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
! ss -ltn|grep -qE ':(8005|18084) '

run_batch_audit(){
 local bid="$1" out="$2" log="$3"
 setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$AUD" --preregistration "$PRE" --selection "$SEL" --collector "$COL" --generator "$GEN" --dataset "$DATASET" --support-root "$SUPPORT" --task-config "$TASK" --resize-source "$RESIZE" --output "$out" --batch-id "$bid" >"$log" 2>&1 & ACTIVE_PID=$!
 local deadline=$((SECONDS+1800))
 while kill -0 "$ACTIVE_PID" 2>/dev/null;do ((SECONDS<deadline))&&within_whole||{ terminate_active;return 124;};sleep 2;done
 wait "$ACTIVE_PID";ACTIVE_PID=''
}

for bid in $(seq 0 9);do
 CURRENT_BATCH=$bid
 within_whole||{ STAGE='whole_timeout';exit 124;}
 AUDREP="$AUDDIR/batch_$(printf '%03d' "$bid")_audit.json"
 if ((bid==0));then
  test -s "$AUDREP" && test "$(sha256sum "$AUDREP"|awk '{print $1}')" = '713c6976a42e817d1f66414b30b35c6e72f505a2a4f68e9a4b778adc8bca04b9'
  STAGE='verify_registered_batch_0';VERIFY="$ATTEMPT/verify_batch_000.json";STAGE_LOG="$ATTEMPT/verify_batch_000.log";run_batch_audit 0 "$VERIFY" "$STAGE_LOG";cmp -s "$VERIFY" "$AUDREP"
  continue
 fi
 if [[ -s "$AUDREP" ]];then
  STAGE="verify_existing_batch_${bid}";VERIFY="$ATTEMPT/verify_batch_$(printf '%03d' "$bid").json";STAGE_LOG="$ATTEMPT/verify_batch_$(printf '%03d' "$bid").log";run_batch_audit "$bid" "$VERIFY" "$STAGE_LOG";cmp -s "$VERIFY" "$AUDREP"
  continue
 fi
 STAGE="collect_batch_${bid}";STAGE_LOG="$ATTEMPT/collect_batch_$(printf '%03d' "$bid").log"
 setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --collector "$COL" --support-root "$SUPPORT" --task-config "$TASK" --output "$DATASET" --batch-id "$bid" >"$STAGE_LOG" 2>&1 & ACTIVE_PID=$!
 BATCH_START=$(date +%s)
 while kill -0 "$ACTIVE_PID" 2>/dev/null;do
  sleep 2
  (( $(date +%s)-BATCH_START <= 5400 ))&&within_whole||{ terminate_active;exit 124;}
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}')||{ terminate_active;exit 125;}
  [[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=24576))||{ terminate_active;exit 125;}
 done
 wait "$ACTIVE_PID";ACTIVE_PID=''
 STAGE="audit_batch_${bid}";STAGE_LOG="$ATTEMPT/audit_batch_$(printf '%03d' "$bid").log";run_batch_audit "$bid" "$AUDREP" "$STAGE_LOG"
done

CURRENT_BATCH=-1;GEN_FINAL_RC=0
if ((GENERATION_PRESENT==0));then
 STAGE='finalize_generation';STAGE_LOG="$ATTEMPT/finalize_generation.log"
 set +e
 setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --collector "$COL" --support-root "$SUPPORT" --task-config "$TASK" --output "$DATASET" --finalize >"$STAGE_LOG" 2>&1 & ACTIVE_PID=$!
 FINALIZE_DEADLINE=$((SECONDS+3600))
 while kill -0 "$ACTIVE_PID" 2>/dev/null;do ((SECONDS<FINALIZE_DEADLINE))&&within_whole||{ terminate_active;GEN_FINAL_RC=124;break;};sleep 2;done
 if [[ -n "$ACTIVE_PID" ]];then wait "$ACTIVE_PID";GEN_FINAL_RC=$?;ACTIVE_PID='';fi
 set -e
fi
((GEN_FINAL_RC==0||GEN_FINAL_RC==2))
STAGE='final_audit';STAGE_LOG="$ATTEMPT/final_audit.log";AUDIT_OUTPUT="$FINAL_AUDIT"
((RECOVER_FINAL==1))&&AUDIT_OUTPUT="$ATTEMPT/recomputed_final_audit.json"
set +e
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$AUD" --preregistration "$PRE" --selection "$SEL" --collector "$COL" --generator "$GEN" --dataset "$DATASET" --support-root "$SUPPORT" --task-config "$TASK" --resize-source "$RESIZE" --output "$AUDIT_OUTPUT" --batch-audit-dir "$AUDDIR" --finalize >"$STAGE_LOG" 2>&1 & ACTIVE_PID=$!
FINAL_DEADLINE=$((SECONDS+3600))
while kill -0 "$ACTIVE_PID" 2>/dev/null;do ((SECONDS<FINAL_DEADLINE))&&within_whole||{ terminate_active;FINAL_RC=124;break;};sleep 2;done
if [[ -n "$ACTIVE_PID" ]];then wait "$ACTIVE_PID";FINAL_RC=$?;ACTIVE_PID='';fi
set -e
if ((RECOVER_FINAL==1));then cmp -s "$AUDIT_OUTPUT" "$FINAL_AUDIT"||FINAL_RC=3;fi
if ((GEN_FINAL_RC!=0||FINAL_RC!=0));then STAGE='terminal_integrity_or_diagnostic_failure';exit 3;fi
within_whole||{ STAGE='whole_timeout_after_final_audit';exit 124;}

STAGE='restore_v218';restore;RESTORED=1
within_whole||{ STAGE='whole_timeout_after_restore';exit 124;}
STAGE='success_receipt';write_attempt_receipt "$INNER_SUCCESS_RECEIPT" 1 0
SUCCESS=1
test ! -e "$SUCCESS_RECEIPT" && cp --reflink=never "$INNER_SUCCESS_RECEIPT" "$SUCCESS_RECEIPT.tmp" && sync -f "$SUCCESS_RECEIPT.tmp" && mv "$SUCCESS_RECEIPT.tmp" "$SUCCESS_RECEIPT" && sync -f "$REG"
trap - EXIT INT TERM
exit 0
