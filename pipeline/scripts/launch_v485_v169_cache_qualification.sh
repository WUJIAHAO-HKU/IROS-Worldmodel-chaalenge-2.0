#!/usr/bin/env bash
# Argument-driven, self-bound one-shot launcher for v485 Phase-A cache qualification.
set -euo pipefail

CONTRACT_SHA_EXPECTED='8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64'
WHOLE_TIMEOUT=9000
TERM_GRACE=15
GPU_LIMIT_MIB=24576
SHM_MIN_BYTES=6442450944
STATIC_CHECK_COUNT=47
STATIC_CHECK_KEYSET_SHA='57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b'

die(){ printf '%s\n' "$1" >&2; exit "${2:-78}"; }
sha_file(){ sha256sum -- "$1" | awk '{print $1}'; }
need_hex(){ [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }
need_regular(){ [[ -f "$1" && ! -L "$1" ]]; }

# Zero-state smoke is before argument parsing, flock, directory creation and imports.
if [[ "${1:-}" == '--synthetic-self-test' ]]; then
  [[ $# -eq 1 ]] || exit 78
  [[ "$CONTRACT_SHA_EXPECTED" =~ ^[0-9a-f]{64}$ ]]
  grep -q 'exec 9>' "$0" && grep -q 'trap cleanup EXIT' "$0"
  grep -q -- '--preregistration' "$0" && grep -q -- '--authority-sha' "$0"
  grep -q 'query-compute-apps=pid' "$0"
  printf '%s\n' '{"format":"strict-track2-v485-launcher-zero-state-synthetic-v1","passed":true}'
  exit 0
fi

PRE=''; PRE_SHA=''; AUTHORITY=''; AUTHORITY_SHA=''; CONTRACT=''; CONTRACT_SHA=''
SCOPE=''; SCOPE_SHA=''; WORKER=''; WORKER_SHA=''; DRIVER=''; DRIVER_SHA=''
AUDITOR=''; AUDITOR_SHA=''; MATERIALIZER=''; MATERIALIZER_SHA=''
STATIC_AUDITOR=''; STATIC_AUDITOR_SHA=''
RESTART_SOURCE=''; RESTART_SHA=''
AUTHORITY_CONTRACT=''; AUTHORITY_CONTRACT_SHA=''
while (($#)); do
  case "$1" in
    --preregistration) PRE="${2:-}"; shift 2;; --prereg-sha) PRE_SHA="${2:-}"; shift 2;;
    --authority) AUTHORITY="${2:-}"; shift 2;; --authority-sha) AUTHORITY_SHA="${2:-}"; shift 2;;
    --contract) CONTRACT="${2:-}"; shift 2;; --contract-sha) CONTRACT_SHA="${2:-}"; shift 2;;
    --scope-source) SCOPE="${2:-}"; shift 2;; --scope-sha) SCOPE_SHA="${2:-}"; shift 2;;
    --worker-source) WORKER="${2:-}"; shift 2;; --worker-sha) WORKER_SHA="${2:-}"; shift 2;;
    --driver-source) DRIVER="${2:-}"; shift 2;; --driver-sha) DRIVER_SHA="${2:-}"; shift 2;;
    --auditor-source) AUDITOR="${2:-}"; shift 2;; --auditor-sha) AUDITOR_SHA="${2:-}"; shift 2;;
    --materializer-source) MATERIALIZER="${2:-}"; shift 2;; --materializer-sha) MATERIALIZER_SHA="${2:-}"; shift 2;;
    --static-auditor-source) STATIC_AUDITOR="${2:-}"; shift 2;; --static-auditor-sha) STATIC_AUDITOR_SHA="${2:-}"; shift 2;;
    --restart-source) RESTART_SOURCE="${2:-}"; shift 2;; --restart-sha) RESTART_SHA="${2:-}"; shift 2;;
    --authority-contract) AUTHORITY_CONTRACT="${2:-}"; shift 2;; --authority-contract-sha) AUTHORITY_CONTRACT_SHA="${2:-}"; shift 2;;
    *) die "unknown or incomplete launcher argument: $1" 78;;
  esac
done

# Everything through the resource/health checks is read-only. Missing arguments
# and all closure failures therefore leave no lock, receipt, or output directory.
for value in "$PRE" "$AUTHORITY" "$CONTRACT" "$SCOPE" "$WORKER" "$DRIVER" "$AUDITOR" "$MATERIALIZER" "$STATIC_AUDITOR" "$RESTART_SOURCE" "$AUTHORITY_CONTRACT"; do
  [[ -n "$value" && "$value" == /* ]] || die 'missing/nonabsolute launcher path argument' 78
  need_regular "$value" || die "launcher input is not regular/nonsymlink: $value" 78
done
for value in "$PRE_SHA" "$AUTHORITY_SHA" "$CONTRACT_SHA" "$SCOPE_SHA" "$WORKER_SHA" "$DRIVER_SHA" "$AUDITOR_SHA" "$MATERIALIZER_SHA" "$STATIC_AUDITOR_SHA" "$RESTART_SHA" "$AUTHORITY_CONTRACT_SHA"; do need_hex "$value" || die 'missing/malformed launcher SHA argument' 78; done
[[ "$CONTRACT_SHA" == "$CONTRACT_SHA_EXPECTED" ]] || die 'wrong v485 Phase-A contract SHA' 78
for pair in "$PRE:$PRE_SHA" "$AUTHORITY:$AUTHORITY_SHA" "$CONTRACT:$CONTRACT_SHA" "$SCOPE:$SCOPE_SHA" "$WORKER:$WORKER_SHA" "$DRIVER:$DRIVER_SHA" "$AUDITOR:$AUDITOR_SHA" "$MATERIALIZER:$MATERIALIZER_SHA" "$STATIC_AUDITOR:$STATIC_AUDITOR_SHA" "$RESTART_SOURCE:$RESTART_SHA" "$AUTHORITY_CONTRACT:$AUTHORITY_CONTRACT_SHA"; do
  path=${pair%:*}; expected=${pair##*:}; [[ "$(sha_file "$path")" == "$expected" ]] || die "input hash mismatch: $path" 78
done
LAUNCHER=$(readlink -f -- "$0"); need_regular "$LAUNCHER" || die 'bad launcher self path' 78
LAUNCHER_SHA=$(sha_file "$LAUNCHER"); REG=$(dirname -- "$PRE")
[[ -d "$REG" && ! -L "$REG" ]] || die 'bad registration root' 78

RLPY=$(python3 - "$PRE" <<'PY'
import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['execution_interpreter']['lexical_path'])
PY
) || die 'cannot extract execution interpreter' 78
[[ "$RLPY" == /* && -e "$RLPY" ]] || die 'bad execution interpreter lexical path' 78
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(dirname "$DRIVER"):${PYTHONPATH:-}"

"$RLPY" - "$PRE" "$PRE_SHA" "$AUTHORITY" "$AUTHORITY_SHA" "$CONTRACT" "$CONTRACT_SHA" \
 "$SCOPE" "$SCOPE_SHA" "$WORKER" "$WORKER_SHA" "$DRIVER" "$DRIVER_SHA" "$AUDITOR" "$AUDITOR_SHA" \
 "$MATERIALIZER" "$MATERIALIZER_SHA" "$STATIC_AUDITOR" "$STATIC_AUDITOR_SHA" "$LAUNCHER" "$LAUNCHER_SHA" "$RESTART_SOURCE" "$RESTART_SHA" "$STATIC_CHECK_COUNT" "$STATIC_CHECK_KEYSET_SHA" "$AUTHORITY_CONTRACT" "$AUTHORITY_CONTRACT_SHA" <<'PY'
import hashlib,json,os,platform,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def regular(p,s):
 p=Path(p);return p.is_file() and not p.is_symlink() and sha(p)==s
a=sys.argv[1:];pre_path,pre_sha,auth_path,auth_sha,contract_path,contract_sha=a[:6];pairs=list(zip(a[6:20:2],a[7:20:2]));restart_path,restart_sha=a[20:22];static_check_count=int(a[22]);static_keyset_sha=a[23];authority_contract_path,authority_contract_sha=a[24:26]
pre=json.loads(Path(pre_path).read_text());auth=json.loads(Path(auth_path).read_text())
assert sha(pre_path)==pre_sha and sha(auth_path)==auth_sha and sha(contract_path)==contract_sha
assert pre['format']=='strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1'
assert pre['status']=='preregistered_cache_qualification_pending_postregistration_static_authority'
assert pre['contract']=={'path':str(Path(contract_path).resolve()),'sha256':contract_sha}
assert pre['authorization']=={'phase_a_cache_qualification_authorized':False,'attempts_authorized':0,'cache_reuse_authorized':False,'training_authorized':False,'folds_authorized':0,'policy_updates':0,'s1_authorized':False,'zero_update_authorized':False,'rl_authorized':False,'submission_authorized':False}
assert pre['required_output_absence_at_registration'] is True and not os.path.lexists(pre['qualification_output_root'])
assert auth['format']=='strict-track2-v485-v169-cache-qualification-postregistration-authority-v1'
assert auth['status']=='authorized_exact_one_phase_a_cache_qualification_attempt' and auth['passed'] is True
assert Path(auth['preregistration_path']).resolve()==Path(pre_path).resolve() and auth['preregistration_sha256']==pre_sha
assert Path(auth['contract_path']).resolve()==Path(contract_path).resolve() and auth['contract_sha256']==contract_sha
assert Path(auth['qualification_output_root']).resolve()==Path(pre['qualification_output_root']).resolve()
assert auth['phase_a_cache_qualification_authorized'] is True and auth['attempts_authorized']==1
for k,v in {'cache_reuse_authorized':False,'training_authorized':False,'folds_authorized':0,'policy_updates':0,'s1_authorized':False,'zero_update_authorized':False,'rl_authorized':False,'submission_authorized':False,'retry_authorized':False}.items():assert auth[k]==v
restart=auth['restart_v218_source'];rp=Path(restart_path).resolve();assert regular(rp,restart_sha) and restart=={'path':str(rp),'sha256':restart_sha,'logical_bytes':rp.stat().st_size}
acp=Path(authority_contract_path).resolve();assert regular(acp,authority_contract_sha) and auth['authority_design_contract']=={'path':str(acp),'sha256':authority_contract_sha,'logical_bytes':acp.stat().st_size}
input_roles=('phase_a_cache_scope_helper','phase_a_process_worker','phase_a_driver','phase_a_independent_auditor','phase_a_materializer','phase_a_static_auditor','phase_a_launcher')
formal_roles=('phase_a_materializer','phase_a_driver','phase_a_process_worker','phase_a_cache_scope_helper','phase_a_independent_auditor','phase_a_static_auditor','phase_a_launcher')
assert len(pairs)==7 and set(pre['execution_sources'])==set(formal_roles);actual={}
for role,(path,want) in zip(input_roles,pairs):
 p=Path(path).resolve();assert regular(p,want);rec=pre['execution_sources'][role]
 assert Path(rec['path']).resolve()==p and rec['sha256']==want and rec['logical_bytes']==p.stat().st_size
 actual[role]={'path':str(p),'sha256':want,'logical_bytes':p.stat().st_size}
records=[{'role':role,**actual[role]} for role in formal_roles]
assert pre['execution_source_records']==records
assert pre['execution_sources_digest_sha256']==hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
st=auth['static_audit'];sp=Path(st['path']);assert regular(sp,st['sha256'])
s=json.loads(sp.read_text());assert s['format']=='strict-track2-v485-v169-cache-qualification-static-audit-v1' and s['passed'] is True and s['status']=='passed_no_execution_authority'
assert s['preregistration']=={'path':str(Path(pre_path).resolve()),'sha256':pre_sha} and s['static_auditor_self_sha256']==pairs[5][1]
assert s['contract']=={'path':str(Path(contract_path).resolve()),'sha256':contract_sha}
assert isinstance(s['checks'],dict) and s['checks'] and set(s['checks'].values())=={True}
check_keys=sorted(s['checks']);assert len(check_keys)==static_check_count==47 and static_keyset_sha=='57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b'
assert s['check_keys']==check_keys and s['check_key_set_sha256']==static_keyset_sha==hashlib.sha256(json.dumps(check_keys,separators=(',',':')).encode()).hexdigest() and s['checks_sha256']==hashlib.sha256(json.dumps(s['checks'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert st['check_count']==static_check_count and st['check_key_set_sha256']==static_keyset_sha and st['checks_sha256']==s['checks_sha256']
assert set(s['sources'])==set(formal_roles)
static_records=[]
for role in formal_roles:
 rec=s['sources'][role];src=pre['execution_sources'][role];assert rec==src and regular(rec['path'],rec['sha256']);static_records.append({'role':role,**rec})
assert s['sources_digest_sha256']==hashlib.sha256(json.dumps(static_records,sort_keys=True,separators=(',',':')).encode()).hexdigest()==pre['execution_sources_digest_sha256']
assert s['runtime_observation']=={'phase_a_executed':False,'training_launched':False,'folds':0,'policy_updates':0,'rl_authorized':False}
assert s['phase_a_cache_qualification_authorized'] is False and s['training_authorized'] is False and s['submission_authorized'] is False
i=pre['execution_interpreter'];lex=Path(i['lexical_path'])
assert lex==Path(sys.executable) and lex.is_symlink() and i['lexical_is_symlink'] is True and os.readlink(lex)==i['symlink_target']
mid=Path(i['symlink_target']);assert mid.is_symlink() and os.readlink(mid)=='python3.11'
resolved=lex.resolve(strict=True);assert str(resolved)==i['resolved_path'] and regular(resolved,i['resolved_sha256']) and resolved.stat().st_size==i['resolved_bytes']
import numpy,torch
assert platform.python_version()==i['python_version'] and numpy.__version__==i['numpy_version'] and torch.__version__==i['torch_version']
assert os.environ['PYTHONHASHSEED']=='0' and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
PY

QUAL_ROOT=$("$RLPY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["qualification_output_root"])' "$PRE")
[[ "$QUAL_ROOT" == /* && ! -e "$QUAL_ROOT" ]] || die 'qualification root must be absent/absolute' 78
[[ "$(dirname "$QUAL_ROOT")" != "$REG" && "$QUAL_ROOT" != "$REG"/* && "$REG" != "$QUAL_ROOT"/* ]] || die 'root containment violation' 78
SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1);[[ "$SHM_AVAIL" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || die 'shm query failed' 78
(( SHM_AVAIL>=SHM_MIN_BYTES )) || die 'insufficient shm' 78
[[ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8005/v1/health 2>/dev/null || true)" == 200 ]] || die 'bridge unhealthy' 78
[[ "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:18084/health 2>/dev/null || true)" == 200 ]] || die 'gpu service unhealthy' 78

ATTEMPTS="$REG/phase_a_launcher_attempts";ATTEMPT="$ATTEMPTS/attempt_000";ATTEMPT_PREP="$ATTEMPTS/.attempt_000.prep"
for p in "$ATTEMPTS" "$ATTEMPT_PREP";do [[ ! -e "$p" ]]||die "one-shot state exists: $p" 78;done

# First mutation: lock, then whole-directory atomic launcher intent.
exec 9>/var/lock/v485_v169_cache_qualification.lock
flock -n 9 || exit 73
START_EPOCH=$(date +%s);ACTIVE_PID='';ACTIVE_PGID='';STAGE='attempt_intent';V218_STOPPED=0;RESTORED=0
GROUP_EMPTY=1;GPU_EMPTY=1;GPU_PEAK_MIB=0;TERMINAL_SUCCESS=0;H8005=0;H18084=0
early_cleanup(){
  local rc=$?;set +e
  mkdir -p "$ATTEMPTS"
  if [[ -d "$ATTEMPT_PREP" && ! -e "$ATTEMPT" ]];then mv -T "$ATTEMPT_PREP" "$ATTEMPT";fi
  [[ -d "$ATTEMPT" ]]||mkdir "$ATTEMPT"
  if [[ ! -e "$ATTEMPT/failure_receipt.json" ]];then
    V485_EARLY_RC="$rc" V485_EARLY_STAGE="$STAGE" "$RLPY" - "$ATTEMPT/failure_receipt.json" "$LAUNCHER" "$LAUNCHER_SHA" "$PRE" "$PRE_SHA" "$AUTHORITY" "$AUTHORITY_SHA" "$AUTHORITY_CONTRACT" "$AUTHORITY_CONTRACT_SHA" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp');rows=[]
for q in sorted(p.parent.rglob('*')):
 if q.is_symlink():raise RuntimeError('early attempt symlink')
 if q.is_file():rows.append([q.relative_to(p.parent).as_posix(),sha(q),q.stat().st_size])
o={'format':'strict-track2-v485-v169-cache-qualification-launcher-terminal-v1','passed':False,'stage':os.environ['V485_EARLY_STAGE'],'exit_code':int(os.environ['V485_EARLY_RC']),'launcher_path':str(Path(sys.argv[2]).resolve()),'launcher_sha256':sys.argv[3],'preregistration_path':str(Path(sys.argv[4]).resolve()),'preregistration_sha256':sys.argv[5],'authority_path':str(Path(sys.argv[6]).resolve()),'authority_sha256':sys.argv[7],'authority_contract_path':str(Path(sys.argv[8]).resolve()),'authority_contract_sha256':sys.argv[9],'attempt_intent_committed':(p.parent/'intent.json').is_file(),'pre_failure_attempt_files':rows,'pre_failure_attempt_tree_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),'v218_was_stopped':False,'v218_health_was_not_mutated':True,'retry_authorized':False,'cache_reuse_authorized':False,'training_authorized':False,'folds_authorized':0,'policy_updates':0,'s1_authorized':False,'zero_update_authorized':False,'rl_authorized':False,'submission_authorized':False}
with tmp.open('x') as f:json.dump(o,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd);fd=os.open(str(p.parent.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
  fi
}
trap early_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$ATTEMPTS";mkdir "$ATTEMPT_PREP"
"$RLPY" - "$ATTEMPT_PREP/intent.json" "$LAUNCHER" "$LAUNCHER_SHA" "$PRE" "$PRE_SHA" "$AUTHORITY" "$AUTHORITY_SHA" "$QUAL_ROOT" "$START_EPOCH" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);o={'format':'strict-track2-v485-v169-cache-qualification-launcher-attempt-intent-v1','launcher_path':str(Path(sys.argv[2]).resolve()),'launcher_sha256':sys.argv[3],'preregistration_path':str(Path(sys.argv[4]).resolve()),'preregistration_sha256':sys.argv[5],'authority_path':str(Path(sys.argv[6]).resolve()),'authority_sha256':sys.argv[7],'qualification_output_root':str(Path(sys.argv[8]).resolve()),'start_epoch_seconds':int(sys.argv[9]),'one_shot':True,'retry_authorized':False,'training_authorized':False,'policy_updates':0,'rl_authorized':False,'submission_authorized':False}
with p.open('x') as f:json.dump(o,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
mv -T "$ATTEMPT_PREP" "$ATTEMPT";"$RLPY" -c 'import os,sys;fd=os.open(sys.argv[1],os.O_RDONLY);os.fsync(fd);os.close(fd)' "$ATTEMPTS"
STAGE_LOG="$ATTEMPT/stage.log";RESTORE_LOG="$ATTEMPT/restore_v218.log";GPU_EVIDENCE="$ATTEMPT/preworker_gpu_evidence.json";: >"$STAGE_LOG"

whole_ok(){ (( $(date +%s)-START_EPOCH<=WHOLE_TIMEOUT )); }
gpu_ok(){ local raw used;raw=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null)||return 1;used=$(awk 'NF{s+=$1}END{print s+0}'<<<"$raw");[[ "$used" =~ ^[0-9]+$ ]]||return 1;((used>GPU_PEAK_MIB))&&GPU_PEAK_MIB=$used;((used<=GPU_LIMIT_MIB)); }
terminate_active(){
 local pid="${ACTIVE_PID:-}" pgid="${ACTIVE_PGID:-}"
 if [[ -n "$pgid" ]]&&kill -0 -- "-$pgid" 2>/dev/null;then kill -TERM -- "-$pgid" 2>/dev/null||true;for _ in $(seq 1 "$TERM_GRACE");do kill -0 -- "-$pgid" 2>/dev/null||break;sleep 1;done;kill -KILL -- "-$pgid" 2>/dev/null||true;fi
 [[ -n "$pid" ]]&&wait "$pid" 2>/dev/null||true
 if [[ -n "$pgid" ]]&&kill -0 -- "-$pgid" 2>/dev/null;then GROUP_EMPTY=0;fi
 ACTIVE_PID='';ACTIVE_PGID=''
}
health_codes(){ H8005=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8005/v1/health 2>/dev/null||true);H18084=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:18084/health 2>/dev/null||true);[[ "$H8005" =~ ^[0-9]+$ ]]||H8005=0;[[ "$H18084" =~ ^[0-9]+$ ]]||H18084=0; }
restore_v218(){
 terminate_active
 if ((V218_STOPPED==1));then local pids;pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null|awk '$1~/^[0-9]+$/{print $1}')||GPU_EMPTY=0;[[ -z "$pids" ]]||GPU_EMPTY=0;bash "$RESTART_SOURCE" start >>"$RESTORE_LOG" 2>&1||true;fi
 for _ in $(seq 1 120);do health_codes;if ((H8005==200&&H18084==200));then RESTORED=1;return 0;fi;sleep 1;done;RESTORED=0;return 1
}
write_receipt(){
 local out="$1" passed="$2" rc="$3";[[ ! -e "$out"&&! -e "$out.tmp" ]]||return 1
 V485_STAGE="$STAGE" V485_PASSED="$passed" V485_RC="$rc" V485_START="$START_EPOCH" V485_RESTORED="$RESTORED" V485_H8005="$H8005" V485_H18084="$H18084" V485_GROUP_EMPTY="$GROUP_EMPTY" V485_GPU_EMPTY="$GPU_EMPTY" V485_GPU_PEAK="$GPU_PEAK_MIB" "$RLPY" - "$out" "$LAUNCHER" "$PRE" "$AUTHORITY" "$AUTHORITY_CONTRACT" "$CONTRACT" "$RESTART_SOURCE" "$QUAL_ROOT" "$ATTEMPT" "$STAGE_LOG" "$RESTORE_LOG" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ev(s):
 p=Path(s)
 if p.is_file() and not p.is_symlink():return {'kind':'file','path':str(p.resolve()),'sha256':sha(p),'logical_bytes':p.stat().st_size}
 if p.is_dir() and not p.is_symlink():
  rows=[]
  for q in sorted(p.rglob('*')):
   if q.is_symlink():raise RuntimeError('terminal symlink')
   if q.is_file():rows.append([q.relative_to(p).as_posix(),sha(q),q.stat().st_size])
  return {'kind':'directory','path':str(p.resolve()),'file_count':len(rows),'logical_file_bytes':sum(r[2] for r in rows),'canonical_tree_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()}
 return {'kind':'absent','path':str(p)}
out=Path(sys.argv[1]);o={'format':'strict-track2-v485-v169-cache-qualification-launcher-terminal-v1','passed':os.environ['V485_PASSED']=='1','stage':os.environ['V485_STAGE'],'exit_code':int(os.environ['V485_RC']),'wall_seconds':max(0,time.time()-int(os.environ['V485_START'])),'launcher':ev(sys.argv[2]),'preregistration':ev(sys.argv[3]),'authority':ev(sys.argv[4]),'authority_design_contract':ev(sys.argv[5]),'contract':ev(sys.argv[6]),'restart_v218_source':ev(sys.argv[7]),'qualification_output':ev(sys.argv[8]),'attempt':ev(sys.argv[9]),'stage_log':ev(sys.argv[10]),'restore_log':ev(sys.argv[11]),'gpu_peak_mib':int(os.environ['V485_GPU_PEAK']),'process_group_empty_before_restore':os.environ['V485_GROUP_EMPTY']=='1','gpu_compute_pids_empty_before_restore':os.environ['V485_GPU_EMPTY']=='1','v218_health_restored':os.environ['V485_RESTORED']=='1','v218_health':{'8005_v1_health_http_code':int(os.environ['V485_H8005']),'18084_health_http_code':int(os.environ['V485_H18084'])},'retry_authorized':False,'cache_reuse_authorized':False,'training_authorized':False,'folds_authorized':0,'policy_updates':0,'s1_authorized':False,'zero_update_authorized':False,'rl_authorized':False,'submission_authorized':False}
tmp=out.with_name(out.name+'.tmp')
with tmp.open('x') as f:json.dump(o,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,out);fd=os.open(str(out.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
cleanup(){ local rc=$?;terminate_active;restore_v218||true;if [[ -s "$ATTEMPT/success_receipt.json" ]];then TERMINAL_SUCCESS=1;fi;if ((TERMINAL_SUCCESS==0));then write_receipt "$ATTEMPT/failure_receipt.json" 0 "$rc"||true;fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STAGE='stop_v218';for name in wm_v218_bridge wm_v218_gpu;do screen -S "$name" -X quit >/dev/null 2>&1||true;done
for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
! ss -ltn|grep -qE ':(8005|18084) ';V218_STOPPED=1
set +e;PID_RAW=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1);PID_RC=$?;MEM_RAW=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>&1);MEM_RC=$?;set -e
V485_PID_RAW="$PID_RAW" V485_PID_RC="$PID_RC" V485_MEM_RAW="$MEM_RAW" V485_MEM_RC="$MEM_RC" "$RLPY" - "$GPU_EVIDENCE" <<'PY'
import json,os,re,sys
from pathlib import Path
def nums(k):
 a=[x.strip() for x in os.environ[k].splitlines() if x.strip()]
 if any(not re.fullmatch(r'[0-9]+',x) for x in a):raise ValueError(k)
 return [int(x) for x in a]
p=Path(sys.argv[1]);pr=int(os.environ['V485_PID_RC']);mr=int(os.environ['V485_MEM_RC'])
try:pids=nums('V485_PID_RAW');mem=nums('V485_MEM_RAW');schema=True
except ValueError:pids=[];mem=[];schema=False
passed=pr==0 and mr==0 and schema and pids==[] and mem==[]
with p.open('x') as f:json.dump({'format':'strict-track2-v485-preworker-gpu-empty-evidence-v1','passed':passed,'pid_query_exit_code':pr,'memory_query_exit_code':mr,'compute_pids':pids,'used_memory_mib':mem,'numeric_schema':schema},f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd);raise SystemExit(0 if passed else 125)
PY

STAGE='phase_a_driver'
setsid env PYTHONPATH="$PYTHONPATH" taskset -c 0-11 "$RLPY" "$DRIVER" --preregistration "$PRE" --preregistration-sha "$PRE_SHA" --contract "$CONTRACT" --authority-receipt "$AUTHORITY" --authority-receipt-sha "$AUTHORITY_SHA" --scope-source "$SCOPE" --scope-sha "$SCOPE_SHA" --worker-source "$WORKER" --worker-sha "$WORKER_SHA" --auditor-source "$AUDITOR" --auditor-sha "$AUDITOR_SHA" --materializer-source "$MATERIALIZER" --materializer-sha "$MATERIALIZER_SHA" --static-auditor-source "$STATIC_AUDITOR" --static-auditor-sha "$STATIC_AUDITOR_SHA" --launcher-source "$LAUNCHER" --launcher-sha "$LAUNCHER_SHA" >>"$STAGE_LOG" 2>&1 & ACTIVE_PID=$!;ACTIVE_PGID=$ACTIVE_PID
while kill -0 "$ACTIVE_PID" 2>/dev/null;do whole_ok||{ STAGE='whole_timeout';terminate_active;exit 124;};gpu_ok||{ STAGE='gpu_resource_violation';terminate_active;exit 125;};sleep 2;done
set +e;wait "$ACTIVE_PID";DRIVER_RC=$?;set -e;terminate_active
((DRIVER_RC==0))||{ STAGE='driver_failure';exit "$DRIVER_RC"; }
[[ -s "$QUAL_ROOT/terminal_receipt.json"&&! -e "$QUAL_ROOT/failure_receipt.json" ]]||{ STAGE='driver_terminal_missing';exit 3; }
STAGE='restore_v218';restore_v218;((GROUP_EMPTY==1&&GPU_EMPTY==1&&RESTORED==1&&H8005==200&&H18084==200));whole_ok
STAGE='complete';write_receipt "$ATTEMPT/success_receipt.json" 1 0;TERMINAL_SUCCESS=1
trap - EXIT INT TERM
exit 0
