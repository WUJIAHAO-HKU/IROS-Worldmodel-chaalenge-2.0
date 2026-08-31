#!/usr/bin/env bash
# Post-registration one-shot v482 public-train temporal S0 launcher template.
# Postregistration launcher: one-way bound to canonical r3 preregistration.
# Superseded a707/043 formals were never executed. S1/zero/RL stay unauthorized.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v482_temporal8_residual_s0_r3_seed1624_20260824"
PRE="$REG/preregistration.json"
PRE_SHA='426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881'
CONTRACT="$S/v482_temporal_film_residual_model_design_contract.json"
RUNTIME="$ROOT/pipeline/wam_pipeline/v482_temporal8_residual_runtime.py"
TRAINER="$S/train_v482_temporal8_residual_5fold.py"
PREPARE="$S/prepare_v482_temporal8_residual_s0.py"
AUDITOR="$S/audit_v482_temporal8_residual_s0.py"
PACKAGER="$S/package_v482_temporal8_residual_release.py"
RELEASE_AUDITOR="$S/audit_v482_temporal8_residual_release.py"
STATIC_AUDITOR="$S/audit_v482_temporal8_residual_static.py"
STATIC_AUDITOR_SHA='088447774ad4ae0b67ec3470e4170ca5c8527e5fdfd0ee12fc000549bf785a2e'
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RESULT="$REG/s0_result"
S0_AUDIT="$REG/s0_audit.json"
STATIC_AUDIT="$REG/static_audit.json"
RELEASE="$J/v482_temporal8_residual_release_seed1624_20260824"
RELEASE_AUDIT="$REG/release_audit.json"
SUCCESS_RECEIPT="$REG/launcher_success_receipt.json"
FAILURE_RECEIPT="$REG/launcher_failure_receipt.json"
RESTORE_LOG="$REG/restore_v218.log"
LAUNCH_LOG="$REG/launcher_stage.log"
ATTEMPT="$REG/launcher_attempt"
PRETRAIN_GPU_EVIDENCE="$ATTEMPT/pretrain_gpu_evidence.json"

CACHE_TIMEOUT=1800
FOLD_TIMEOUT=1200
S0_TIMEOUT=7200
ALL200_TIMEOUT=1200
WHOLE_TIMEOUT=8400
GPU_LIMIT_MIB=28672
SHM_MIN_BYTES=6442450944
RELEASE_MAX_BYTES=536870912

# A draft with an unresolved one-way preregistration binding must not create any
# lock, receipt, restore log, or other state if invoked accidentally.
[[ "$PRE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo 'v482 PRE_SHA pending; launcher is not executable' >&2; exit 78; }
exec 9>/var/lock/v482_temporal8_residual_s0.lock
flock -n 9 || exit 73
START_EPOCH=$(date +%s)
ACTIVE_PID=''
ACTIVE_PGID=''
STAGE='preflight'
SUCCESSFUL=0
RESTORED=0
V218_STOPPED=0
PROCESS_GROUP_EMPTY_BEFORE_RESTORE=1
GPU_PIDS_EMPTY_BEFORE_RESTORE=1
PRETRAIN_GPU_CHECKED=0
PRETRAIN_GPU_EMPTY=0
H8005=0
H18084=0
GPU_PEAK_MIB=0

sha256_file(){ sha256sum "$1" | awk '{print $1}'; }
whole_ok(){ (( $(date +%s)-START_EPOCH <= WHOLE_TIMEOUT )); }
gpu_ok(){
  local used
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'NF{s+=$1}END{print s+0}') || return 1
  [[ "$used" =~ ^[0-9]+$ ]] || return 1
  (( used > GPU_PEAK_MIB )) && GPU_PEAK_MIB=$used
  (( used <= GPU_LIMIT_MIB ))
}
terminate_active(){
  local pid="${ACTIVE_PID:-}" pgid="${ACTIVE_PGID:-${ACTIVE_PID:-}}"
  if [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  if [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null; then PROCESS_GROUP_EMPTY_BEFORE_RESTORE=0; fi
  ACTIVE_PID=''; ACTIVE_PGID=''
}
health_codes(){
  H8005=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8005/v1/health 2>/dev/null || true)
  H18084=$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:18084/health 2>/dev/null || true)
  [[ "$H8005" =~ ^[0-9]+$ ]] || H8005=0
  [[ "$H18084" =~ ^[0-9]+$ ]] || H18084=0
}
restore_v218(){
  terminate_active
  if (( V218_STOPPED==1 )); then
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}') || GPU_PIDS_EMPTY_BEFORE_RESTORE=0
    [[ -z "$gpu_pids" ]] || GPU_PIDS_EMPTY_BEFORE_RESTORE=0
  fi
  if [[ ! -e "$RESTORE_LOG" ]]; then
    bash "$S/restart_v218_services.sh" start >"$RESTORE_LOG" 2>&1 || true
  else
    bash "$S/restart_v218_services.sh" start >>"$RESTORE_LOG" 2>&1 || true
  fi
  for _ in $(seq 1 120); do
    health_codes
    if (( H8005==200 && H18084==200 )); then
      RESTORED=1
      (( PROCESS_GROUP_EMPTY_BEFORE_RESTORE==1 && GPU_PIDS_EMPTY_BEFORE_RESTORE==1 ))
      return
    fi
    sleep 1
  done
  RESTORED=0
  return 1
}
write_terminal_receipt(){
  local path="$1" passed="$2" rc="$3"
  [[ -e "$path" ]] && return 0
  V482_STAGE="$STAGE" V482_PASSED="$passed" V482_RC="$rc" V482_START="$START_EPOCH" \
  V482_RESTORED="$RESTORED" V482_H8005="$H8005" V482_H18084="$H18084" V482_GPU_PEAK="$GPU_PEAK_MIB" \
  V482_V218_STOPPED="$V218_STOPPED" V482_GROUP_EMPTY="$PROCESS_GROUP_EMPTY_BEFORE_RESTORE" V482_GPU_EMPTY="$GPU_PIDS_EMPTY_BEFORE_RESTORE" \
  V482_PRETRAIN_GPU_CHECKED="$PRETRAIN_GPU_CHECKED" V482_PRETRAIN_GPU_EMPTY="$PRETRAIN_GPU_EMPTY" \
  "$RLPY" - "$path" "$0" "$PRE" "$CONTRACT" "$TRAINER" "$AUDITOR" "$PACKAGER" "$RELEASE_AUDITOR" \
    "$STATIC_AUDITOR" "$STATIC_AUDIT" "$RESULT" "$RESULT.partial" "$S0_AUDIT" "$RELEASE" "$RELEASE_AUDIT" "$RESTORE_LOG" "$LAUNCH_LOG" "$ATTEMPT" "$PRETRAIN_GPU_EVIDENCE" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ev(p):
 p=Path(p)
 if p.is_file() and not p.is_symlink(): return {'kind':'file','path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
 if p.is_dir() and not p.is_symlink():
  rows=[]
  for q in sorted(p.rglob('*')):
   if q.is_symlink(): raise RuntimeError('terminal evidence symlink')
   if q.is_file(): rows.append([q.relative_to(p).as_posix(),sha(q),q.stat().st_size])
  return {'kind':'directory','path':str(p.resolve()),'file_count':len(rows),'bytes':sum(x[2] for x in rows),'tree_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()}
 return {'kind':'absent','path':str(p)}
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp')
obj={'format':'strict-track2-v482-temporal8-residual-launcher-terminal-v1','passed':os.environ['V482_PASSED']=='1','stage':os.environ['V482_STAGE'],'exit_code':int(os.environ['V482_RC']),'wall_seconds':max(0,int(__import__('time').time())-int(os.environ['V482_START'])),'launcher':ev(sys.argv[2]),'preregistration':ev(sys.argv[3]),'contract':ev(sys.argv[4]),'trainer':ev(sys.argv[5]),'s0_auditor':ev(sys.argv[6]),'packager':ev(sys.argv[7]),'release_auditor':ev(sys.argv[8]),'static_auditor':ev(sys.argv[9]),'static_audit':ev(sys.argv[10]),'result':ev(sys.argv[11]),'result_partial':ev(sys.argv[12]),'s0_audit':ev(sys.argv[13]),'release':ev(sys.argv[14]),'release_audit':ev(sys.argv[15]),'restore_log':ev(sys.argv[16]),'stage_log':ev(sys.argv[17]),'attempt':ev(sys.argv[18]),'pretrain_gpu_evidence':ev(sys.argv[19]),'gpu_peak_mib':int(os.environ['V482_GPU_PEAK']),'pretrain_gpu_check_performed':os.environ['V482_PRETRAIN_GPU_CHECKED']=='1','pretrain_gpu_compute_pids_empty':os.environ['V482_PRETRAIN_GPU_EMPTY']=='1','v218_was_stopped':os.environ['V482_V218_STOPPED']=='1','process_group_empty_before_restore':os.environ['V482_GROUP_EMPTY']=='1','gpu_compute_pids_empty_before_restore':os.environ['V482_GPU_EMPTY']=='1','v218_health_restored':os.environ['V482_RESTORED']=='1','v218_health':{'8005_v1_health_http_code':int(os.environ['V482_H8005']),'18084_health_http_code':int(os.environ['V482_H18084'])},'retry_authorized':False,'submission_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
if p.exists() or tmp.exists(): raise FileExistsError(p)
with tmp.open('x') as f: json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
cleanup(){
  local rc=$?
  terminate_active
  restore_v218 || true
  if [[ -s "$SUCCESS_RECEIPT" ]]; then SUCCESSFUL=1; fi
  if (( SUCCESSFUL==0 )); then write_terminal_receipt "$FAILURE_RECEIPT" 0 "$rc" || true; fi
}
test -s "$PRE"
test "$(sha256_file "$PRE")" = "$PRE_SHA"
for path in "$CONTRACT" "$RUNTIME" "$TRAINER" "$PREPARE" "$AUDITOR" "$PACKAGER" "$RELEASE_AUDITOR" "$STATIC_AUDITOR"; do
  test -s "$path"; test ! -L "$path"
done
test "$(sha256_file "$STATIC_AUDITOR")" = "$STATIC_AUDITOR_SHA"
test ! -e "$RESULT"; test ! -e "$RESULT.partial"; test ! -e "$S0_AUDIT"; test ! -e "$S0_AUDIT.tmp"
test ! -e "$STATIC_AUDIT"; test ! -e "$STATIC_AUDIT.tmp"; test ! -e "$RELEASE"; test ! -e "$RELEASE.partial"
test ! -e "$RELEASE_AUDIT"; test ! -e "$RELEASE_AUDIT.tmp"; test ! -e "$SUCCESS_RECEIPT"; test ! -e "$FAILURE_RECEIPT"
test ! -e "$ATTEMPT"; test ! -e "$ATTEMPT.prep"; test ! -e "$RESTORE_LOG"; test ! -e "$LAUNCH_LOG"
test ! -e /dev/shm/v482_v169_temporal_cache_seed1624; test ! -e /dev/shm/v482_v169_temporal_cache_seed1624.partial
(( $(df -B1 --output=avail /dev/shm | tail -1) >= SHM_MIN_BYTES ))
PYTHONPYCACHEPREFIX=/dev/shm/v482_pycache "$RLPY" -m py_compile "$RUNTIME" "$TRAINER" "$PREPARE" "$AUDITOR" "$PACKAGER" "$RELEASE_AUDITOR" "$STATIC_AUDITOR"
export PYTHONPATH="$ROOT/pipeline:$S" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
"$RLPY" - "$PRE" "$CONTRACT" "$RUNTIME" "$TRAINER" "$PREPARE" "$AUDITOR" "$PACKAGER" "$RELEASE_AUDITOR" <<'PY'
import hashlib,json,os,platform,sys
from pathlib import Path
import numpy,torch
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pre=json.loads(Path(sys.argv[1]).read_text());src=pre['source']
assert pre['format']=='strict-track2-v482-temporal8-residual-preregistration-v1'
assert pre['status']=='preregistered_public_train_temporal_s0_authorized'
assert pre['guards']=={'public_train_only':True,'training_authorized':True,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False,'dev_reward_success_outcome_final_hidden_used':False,'submission_authorized':False}
for key,arg in zip(('contract','runtime','trainer','prepare','auditor','packager','release_auditor'),sys.argv[2:]):
 p=Path(arg).resolve();assert Path(src[key+'_path']).resolve()==p and src[key+'_sha256']==sha(p)
six=pre['execution_source_six'];alias={'runtime':'runtime','trainer':'trainer','prepare':'prepare','auditor':'s0_auditor','packager':'packager','release_auditor':'release_auditor'}
assert pre['execution_source_six_count']==6 and set(six)==set(alias.values())
assert pre['execution_source_six_digest_sha256']==hashlib.sha256(json.dumps(six,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for old,new in alias.items(): assert Path(src[old+'_path']).resolve()==Path(six[new]['path']).resolve() and src[old+'_sha256']==six[new]['sha256']==sha(six[new]['path'])
authority=pre['launcher_authority_and_timeouts'];timeouts=authority['stage_timeout_seconds'];resources=authority['resource_gates']
assert authority['launcher_is_postregistration_orchestrator'] is True and authority['launcher_sha256_is_not_in_this_preregistration_to_avoid_self_reference'] is True and authority['launcher_must_bind_this_preregistration_path_and_sha256_before_execution'] is True and authority['one_shot_attempt'] is True
assert timeouts=={'v169_cache':1800,'each_fold_all_three_heads':1200,'s0_whole':7200,'all200':1200,'full_chain':8400,'term_to_kill_grace':15}
assert resources=={'cpu_affinity':'0-11','total_threads':6,'minimum_shm_free_bytes':6442450944,'gpu_peak_mib_hard_max':28672,'persistent_release_bytes_hard_max':536870912,'no_oom_retry_or_batch_change':True}
assert pre['v169_cache']['exact_scalar_calls']==1000 and pre['v169_cache']['redundancy_rows_passed']==200 and pre['v169_cache']['redundancy_equality_count']==9600
expected_schedule_keys={'fold','heads','identical_order_for_all_heads','seed','sha256','shape','steps_per_head'}
assert len(pre['schedules'])==5 and [x['fold'] for x in pre['schedules']]==list(range(5))
assert all(set(x)==expected_schedule_keys and x['heads']==['action','context_only','phase_shuffle'] and x['identical_order_for_all_heads'] is True and x['seed']==1624+x['fold'] and x['shape']==[400,2] and x['steps_per_head']==400 and len(x['sha256'])==64 for x in pre['schedules'])
assert len(pre['temporal_contexts'])==200 and all(x['redundancy_identity_gate'] is True and x['redundancy_equality_count']==48 for x in pre['temporal_contexts'])
framework=pre['framework'];interp=framework['execution_interpreter'];lexical=Path(interp['lexical_path'])
assert str(lexical)=='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python' and lexical==Path(sys.executable)
assert lexical.is_symlink() and interp['lexical_is_symlink'] is True and os.readlink(lexical)==interp['symlink_target']=='/root/autodl-tmp/conda_envs/isaacsim51/bin/python'
middle=Path(interp['symlink_target']);assert middle.is_symlink() and os.readlink(middle)=='python3.11'
resolved=lexical.resolve(strict=True);assert str(resolved)==interp['resolved_path']=='/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11'
assert resolved.is_file() and not resolved.is_symlink() and interp['resolved_is_regular_file'] is True and resolved.stat().st_size==interp['resolved_bytes']==25555040 and sha(resolved)==interp['resolved_sha256']=='11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788'
assert platform.python_version()==interp['python_version']=='3.11.15' and numpy.__version__==framework['numpy_version']==interp['numpy_version']=='1.26.4' and torch.__version__==framework['torch_version']==interp['torch_version']=='2.7.0+cu128'
assert framework['pythonhashseed_required']=='0' and os.environ.get('PYTHONHASHSEED')=='0' and framework['cublas_workspace_config_required']==':4096:8' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and framework['torch_deterministic_algorithms_required'] is True
sup=pre['supersession'];assert sup['all_previous_executed'] is False and sup['only_authorized_preregistration']=='this native r3 preregistration after independent audit'
assert sup['native_prepare_author']=={'path':str(Path(sys.argv[5]).resolve()),'sha256':sha(sys.argv[5]),'postprocessing_materializer_used':False}
expected_old=[('a70731adf473582fb63a06e70efc61eb938d2045c2879ebc1a55a4de7c54a4f6','non_execution_interpreter_framework_provenance'),('0431814403f8cace842765b17fbee098427187cdbfcb7cc05ffde986f0ece665','unbound_postprocessing_materializer_provenance')]
assert len(sup['previous_formals'])==2
for entry,(old_sha,reason) in zip(sup['previous_formals'],expected_old):
 old=Path(entry['path']);assert entry['executed'] is False and entry['superseded'] is True and entry['sha256']==old_sha and entry['supersession_reason']==reason and entry['parent_directory_exact_files']==['preregistration.json']
 assert old.is_file() and not old.is_symlink() and sha(old)==old_sha and sorted(x.name for x in old.parent.iterdir())==['preregistration.json']
PY

whole_ok
"$RLPY" - "$ATTEMPT" "$0" "$PRE" "$START_EPOCH" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
target=Path(sys.argv[1]);prep=target.with_name(target.name+'.prep')
if target.exists() or prep.exists(): raise FileExistsError(target)
prep.mkdir()
obj={'format':'strict-track2-v482-temporal8-residual-launcher-attempt-intent-v1','launcher_path':str(Path(sys.argv[2]).resolve()),'launcher_sha256':sha(sys.argv[2]),'preregistration_path':str(Path(sys.argv[3]).resolve()),'preregistration_sha256':sha(sys.argv[3]),'start_epoch_seconds':int(sys.argv[4]),'one_shot':True,'retry_authorized':False,'submission_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
p=prep/'intent.json'
with p.open('x') as f: json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(prep),os.O_RDONLY);os.fsync(fd);os.close(fd);os.replace(prep,target);fd=os.open(str(target.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STAGE='static_interface_audit'
setsid env PYTHONPATH="$PYTHONPATH" taskset -c 0-11 "$RLPY" "$STATIC_AUDITOR" \
  --preregistration "$PRE" --contract "$CONTRACT" --runtime "$RUNTIME" --trainer "$TRAINER" --prepare "$PREPARE" \
  --auditor "$AUDITOR" --packager "$PACKAGER" --release-auditor "$RELEASE_AUDITOR" --output "$STATIC_AUDIT" >"$LAUNCH_LOG" 2>&1 & ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
STATIC_START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  whole_ok && (( $(date +%s)-STATIC_START <= 600 )) || { terminate_active; exit 124; }
  sleep 2
done
wait "$ACTIVE_PID"; terminate_active

STAGE='stop_v218_for_training'
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 90); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
! ss -ltn | grep -qE ':(8005|18084) '
V218_STOPPED=1
set +e
PRETRAIN_PID_RAW=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>&1); PRETRAIN_PID_RC=$?
PRETRAIN_MEMORY_RAW=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>&1); PRETRAIN_MEMORY_RC=$?
set -e
set +e
V482_PID_RAW="$PRETRAIN_PID_RAW" V482_PID_RC="$PRETRAIN_PID_RC" V482_MEMORY_RAW="$PRETRAIN_MEMORY_RAW" V482_MEMORY_RC="$PRETRAIN_MEMORY_RC" \
"$RLPY" - "$PRETRAIN_GPU_EVIDENCE" <<'PY'
import json,os,re,sys
from pathlib import Path
def parse(name):
 text=os.environ[name];lines=[line.strip() for line in text.splitlines() if line.strip()]
 if any(re.fullmatch(r"[0-9]+",line) is None for line in lines): raise ValueError(name)
 return [int(line) for line in lines]
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp')
pid_rc=int(os.environ['V482_PID_RC']);memory_rc=int(os.environ['V482_MEMORY_RC'])
try: pids=parse('V482_PID_RAW');memory=parse('V482_MEMORY_RAW');schema=True
except ValueError: pids=[];memory=[];schema=False
passed=pid_rc==0 and memory_rc==0 and schema and pids==[] and memory==[]
obj={'format':'strict-track2-v482-pretrain-gpu-empty-evidence-v1','pid_query_exit_code':pid_rc,'memory_query_exit_code':memory_rc,'numeric_schema':schema,'compute_pids':pids,'used_memory_mib':memory,'compute_pids_empty':pids==[],'used_memory_entries_empty':memory==[],'passed':passed}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
raise SystemExit(0 if passed else 3)
PY
PRETRAIN_GPU_RC=$?
set -e
PRETRAIN_GPU_CHECKED=1
if (( PRETRAIN_GPU_RC!=0 )); then STAGE='pretrain_gpu_pid_not_empty_or_query_failed'; exit 125; fi
PRETRAIN_GPU_EMPTY=1

V169_RELEASE=$("$RLPY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["v169"]["release_path"])' "$PRE")
V169_LIBRARY=$("$RLPY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["v169"]["library_path"])' "$PRE")
STAGE='train_cache_and_fivefold_s0'
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
  taskset -c 0-11 "$RLPY" "$TRAINER" --preregistration "$PRE" --contract "$CONTRACT" --v169-release "$V169_RELEASE" \
  --v169-library "$V169_LIBRARY" --output-dir "$RESULT" --device cuda >>"$LAUNCH_LOG" 2>&1 & ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
TRAIN_START=$(date +%s); PHASE_START=$TRAIN_START; CACHE_SEEN=0; NEXT_FOLD=0
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
  now=$(date +%s)
  whole_ok && (( now-TRAIN_START <= S0_TIMEOUT )) || { STAGE='s0_or_whole_timeout'; terminate_active; exit 124; }
  gpu_ok || { STAGE='gpu_resource_violation'; terminate_active; exit 125; }
  if (( CACHE_SEEN==0 )); then
    if [[ -s "$RESULT.partial/scalar_v169_temporal_cache_receipt.json" ]]; then CACHE_SEEN=1; PHASE_START=$now
    elif (( now-TRAIN_START > CACHE_TIMEOUT )); then STAGE='cache_timeout'; terminate_active; exit 124
    fi
  else
    while (( NEXT_FOLD<5 )) && [[ -s "$RESULT.partial/fold${NEXT_FOLD}_receipt.json" ]]; do
      NEXT_FOLD=$((NEXT_FOLD+1)); PHASE_START=$now
    done
    if (( NEXT_FOLD<5 && now-PHASE_START>FOLD_TIMEOUT )); then STAGE="fold_${NEXT_FOLD}_timeout"; terminate_active; exit 124; fi
    if (( NEXT_FOLD==5 && now-PHASE_START>ALL200_TIMEOUT )); then STAGE='all200_timeout'; terminate_active; exit 124; fi
  fi
  sleep 2
done
set +e; wait "$ACTIVE_PID"; TRAIN_RC=$?; terminate_active; set -e
(( TRAIN_RC==0 || TRAIN_RC==2 )) || { STAGE='trainer_hard_failure'; exit "$TRAIN_RC"; }
test -s "$RESULT/s0_report.json"

STAGE='independent_s0_audit_before_package'
set +e
setsid env PYTHONPATH="$PYTHONPATH" taskset -c 0-11 "$RLPY" "$AUDITOR" --preregistration "$PRE" --contract "$CONTRACT" \
  --result-dir "$RESULT" --output "$S0_AUDIT" >>"$LAUNCH_LOG" 2>&1 & ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
AUDIT_START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null; do whole_ok && (( $(date +%s)-AUDIT_START<=600 )) || { terminate_active; break; }; sleep 2; done
if [[ -n "$ACTIVE_PID" ]]; then wait "$ACTIVE_PID"; AUDIT_RC=$?; terminate_active; else AUDIT_RC=124; fi
set -e
if (( TRAIN_RC!=0 || AUDIT_RC!=0 )); then STAGE='terminal_s0_gate_failure'; exit 3; fi

STAGE='package_passed_candidate'
setsid env PYTHONPATH="$PYTHONPATH" taskset -c 0-11 "$RLPY" "$PACKAGER" --preregistration "$PRE" --contract "$CONTRACT" \
  --result-dir "$RESULT" --audit "$S0_AUDIT" --runtime "$RUNTIME" --trainer "$TRAINER" --prepare "$PREPARE" --auditor "$AUDITOR" \
  --release-auditor "$RELEASE_AUDITOR" --v169-release "$V169_RELEASE" --v169-library "$V169_LIBRARY" --output-dir "$RELEASE" >>"$LAUNCH_LOG" 2>&1 & ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
PACKAGE_START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null; do whole_ok && (( $(date +%s)-PACKAGE_START<=600 )) || { terminate_active; exit 124; }; sleep 2; done
wait "$ACTIVE_PID"; terminate_active
RELEASE_BYTES=$(du -sb "$RELEASE" | awk '{print $1}'); (( RELEASE_BYTES<=RELEASE_MAX_BYTES ))

STAGE='independent_release_audit'
setsid env PYTHONPATH="$PYTHONPATH" taskset -c 0-11 "$RLPY" "$RELEASE_AUDITOR" --release "$RELEASE" --output "$RELEASE_AUDIT" >>"$LAUNCH_LOG" 2>&1 & ACTIVE_PID=$!
ACTIVE_PGID=$ACTIVE_PID
RELEASE_AUDIT_START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null; do whole_ok && (( $(date +%s)-RELEASE_AUDIT_START<=600 )) || { terminate_active; exit 124; }; sleep 2; done
wait "$ACTIVE_PID"; terminate_active

STAGE='restore_v218'; restore_v218
whole_ok
STAGE='complete'
write_terminal_receipt "$SUCCESS_RECEIPT" 1 0
SUCCESSFUL=1
trap - EXIT INT TERM
echo V482_TEMPORAL_S0_COMPLETE
