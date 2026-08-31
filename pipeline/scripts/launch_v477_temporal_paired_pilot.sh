#!/usr/bin/env bash
# One-shot frozen v477 public-train temporal paired simulator pilot. No training, reward, S1, policy, or RL.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v477_temporal8_pilot_seed1621_r2_20260824"
PRE="$REG/preregistration.json"
COL="$S/collect_v477_temporal_paired.py"
GEN="$S/generate_v477_temporal_paired_pilot.py"
AUD="$S/audit_v477_temporal_paired_pilot.py"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
TASK="$SUPPORT/task_config/demo_clean.yml"
RESIZE="$ROOT/pipeline/wam_pipeline/data.py"
DATASET="$REG/dataset"
GENREP="$DATASET/generation_report.json"
AUDREP="$REG/audit_receipt.json"
FAILREP="$REG/launcher_failure_receipt.json"
GENLOG="$REG/generator.log"
AUDLOG="$REG/auditor.log"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

exec 9>/var/lock/v477_temporal8_pilot.lock
flock -n 9 || exit 73
test -s "$PRE"
test ! -e "$DATASET"
test ! -e "$AUDREP"
test ! -e "$FAILREP"
test ! -e "$GENLOG"
test ! -e "$AUDLOG"
test "$(sha256sum "$PRE"|awk '{print $1}')" = '39e973d44bfab5e986c6a3d4175e65b6d275d4267d71fdb4b8f2771d1c793bdb'
test "$(sha256sum "$COL"|awk '{print $1}')" = 'be90f3dd1272706af6d9bc1f461543c9b399071139a4ea243bd962b4b01f187a'
test "$(sha256sum "$GEN"|awk '{print $1}')" = '2a2947b723739ed35e866c44d978b0060e31af6557a75fc2e6c30a9a5d896f8e'
test "$(sha256sum "$AUD"|awk '{print $1}')" = '4f5533bd1dd4e1ad536759d94645c58fcf7c2657e738e863b9f67c74b9fc1323'
PYTHONPYCACHEPREFIX=/dev/shm/v477_pycache "$RLPY" -m py_compile "$COL" "$GEN" "$AUD"
export PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts"
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
"$RLPY" - "$PRE" "$COL" "$GEN" "$AUD" "$SUPPORT" "$TASK" "$RESIZE" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
p=json.load(open(sys.argv[1]));c=p['execution_closure']
assert p['format']=='strict-track2-v477-public-train-paired-temporal8-pilot-preregistration-v1'
assert p['status']=='preregistered_public_train_paired_temporal8_pilot_authorized'
assert p['contract']['sha256']=='9fbb494f55115786b2b22e77d91c172ec1a52df77cec224083ff6e82391e53a1'
for key,arg in zip(('collector','generator','auditor'),sys.argv[2:5]):
 q=Path(arg).resolve();assert Path(c[key]['path']).resolve()==q and c[key]['sha256']==sha(q)
for key,arg in zip(('support_root','task_config','resize_source'),sys.argv[5:8]):
 assert Path(c[key]['path']).resolve()==Path(arg).resolve()
assert c['task_config']['sha256']==sha(sys.argv[6]) and c['resize_source']['sha256']==sha(sys.argv[7])
assert [(x['episode'],x['start']) for x in p['contexts']]==[(25,70),(49,79),(40,118),(28,121)]
assert p['resources']['workers_exact']==2 and p['resources']['cpu_threads_per_worker']==3
assert p['guards']['success_done_outcome_consumed'] is False and p['guards']['all_fixed_rows_branches_retained'] is True
PY

ACTIVE_PID=''
STAGE='preflight'
SUCCESS=0
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
  kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$ACTIVE_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$ACTIVE_PID" 2>/dev/null || true
  wait "$ACTIVE_PID" 2>/dev/null || true
 fi
 ACTIVE_PID=''
}
restore(){
 terminate_active
 bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
 for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1 && curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1; then return 0; fi
  sleep 1
 done
 return 1
}
write_failure(){
 local rc="$1"
 [[ -e "$FAILREP" ]] && return 0
 V477_STAGE="$STAGE" V477_RC="$rc" "$RLPY" - "$FAILREP" "$PRE" "$GENLOG" "$AUDLOG" "$GENREP" "$AUDREP" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evidence(p):
 p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp')
obj={'format':'strict-track2-v477-temporal8-pilot-launcher-failure-v1','passed':False,'stage':os.environ['V477_STAGE'],'exit_code':int(os.environ['V477_RC']),'preregistration_sha256':sha(sys.argv[2]),'generator_log':evidence(sys.argv[3]),'auditor_log':evidence(sys.argv[4]),'generation_report':evidence(sys.argv[5]),'audit_receipt':evidence(sys.argv[6]),'full200_collection_authorized':False,'parent_training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False,'retry_under_v477_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
on_exit(){ local rc="$1"; terminate_active; restore || true; if ((SUCCESS==0)); then write_failure "$rc" || true; fi; }
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}')
[[ "$USED" =~ ^[0-9]+$ ]] && ((USED<=1024))
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 90); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
! ss -ltn | grep -qE ':(8005|18084) '

STAGE='temporal_paired_generation'
cd "$SUPPORT"
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --collector "$COL" --support-root "$SUPPORT" --task-config "$TASK" --output "$DATASET" >"$GENLOG" 2>&1 & ACTIVE_PID=$!
GEN_START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null; do
 sleep 2
 (( $(date +%s)-GEN_START <= 1800 )) || { terminate_active; exit 124; }
 USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}') || { terminate_active; exit 125; }
 [[ "$USED" =~ ^[0-9]+$ ]] && ((USED<=24576)) || { terminate_active; exit 125; }
done
wait "$ACTIVE_PID"; ACTIVE_PID=''
test -s "$GENREP"
(( $(du -sb "$DATASET" | awk '{print $1}') <= 268435456 ))

STAGE='immutable_audit'
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$AUD" --preregistration "$PRE" --collector "$COL" --dataset "$DATASET" --generation-report "$GENREP" --support-root "$SUPPORT" --task-config "$TASK" --resize-source "$RESIZE" --output "$AUDREP" >"$AUDLOG" 2>&1 & ACTIVE_PID=$!
for _ in $(seq 1 300); do kill -0 "$ACTIVE_PID" 2>/dev/null || break; sleep 1; done
kill -0 "$ACTIVE_PID" 2>/dev/null && { terminate_active; exit 124; }
wait "$ACTIVE_PID"; ACTIVE_PID=''
test -s "$AUDREP"

STAGE='restore_v218'
restore
SUCCESS=1
trap - EXIT INT TERM
echo V477_TEMPORAL_PILOT_COMPLETE
