#!/usr/bin/env bash
# One-shot v478 Phase-A: collect only the twelve newly required endpoint oracles.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
S="$ROOT/pipeline/scripts"
REG="$J/v478_endpoint_oracle12_seed1622_20260824"
PRE="$REG/preregistration_r2.json"
OLD_PRE="$REG/preregistration.json"
SEL="$J/v478_temporal200_seed1622_20260824/selection.json"
ORACLE='/root/v478_endpoint_oracle12_seed1622_20260824'
COL="$S/collect_v460_endpoint_only.py"
GEN="$S/generate_v478_endpoint_oracle12.py"
AUD="$S/audit_v478_endpoint_oracle12.py"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
TASK="$SUPPORT/task_config/demo_clean.yml"
RESIZE="$ROOT/pipeline/wam_pipeline/data.py"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GENREP="$ORACLE/generation_report.json"
AUDREP="$REG/audit_receipt.json"
SUCCESSREP="$REG/launcher_receipt.json"
FAILREP="$REG/launcher_failure_receipt.json"
GENLOG="$REG/generator.log"
AUDLOG="$REG/auditor.log"

exec 9>/var/lock/v478_endpoint_oracle12.lock
flock -n 9 || exit 73
test -s "$PRE";test -s "$OLD_PRE";test -s "$SEL";test ! -e "$ORACLE";test ! -e "$AUDREP";test ! -e "$SUCCESSREP";test ! -e "$FAILREP";test ! -e "$GENLOG";test ! -e "$AUDLOG"
test "$(sha256sum "$PRE"|awk '{print $1}')" = '167841e7c0d6a3380a4f524d186bcb9db37c7cf2a73550f711b35cffa547e6b9'
test "$(sha256sum "$OLD_PRE"|awk '{print $1}')" = '201af06ea38b74fd13f63392a2f88cacc6776f1a8193deb584ce9f443d612c3d'
test "$(sha256sum "$SEL"|awk '{print $1}')" = 'f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398'
test "$(sha256sum "$COL"|awk '{print $1}')" = '6c64e4a42f273f43b9e64523bf12e4dc5f24618d5673848db6c1582b5547e846'
test "$(sha256sum "$GEN"|awk '{print $1}')" = 'dfc59e1250251be9f657f260de1c84c1936ce1aec5d6cd67951ed48bd48c86da'
test "$(sha256sum "$AUD"|awk '{print $1}')" = '6cb920fbccfa039ac3a12839ee2626614a6498397adb563c3b91cd5c573cb4a2'
PYTHONPYCACHEPREFIX=/dev/shm/v478_oracle12_pycache "$RLPY" -m py_compile "$COL" "$GEN" "$AUD"
export PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
"$RLPY" - "$PRE" "$SEL" "$COL" "$GEN" "$AUD" "$SUPPORT" "$TASK" "$RESIZE" "$OLD_PRE" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));sel=json.load(open(sys.argv[2]));c=pre['execution_closure']
assert pre['format']=='strict-track2-v478-endpoint-oracle12-preregistration-v1' and pre['status']=='preregistered_public_train_endpoint_oracle_collection_authorized'
master_path=Path(pre['master_contract']['path']).resolve();master=json.load(open(master_path));assert sha(master_path)==pre['master_contract']['sha256']
assert master['format']=='strict-track2-v478-public-train-temporal200-collection-contract-v1' and master['status']=='phase_a_endpoint_oracle12_authorized_phase_b_temporal_false' and master['phase_b_temporal200']['authorized'] is False and master['guards']['phase_b_temporal_collection_authorized'] is False
receipt_path=Path(pre['selection']['receipt_path']).resolve();receipt=json.load(open(receipt_path));assert sha(receipt_path)==pre['selection']['receipt_sha256'] and receipt['format']=='strict-track2-v478-window-eligible-selection-receipt-v1' and receipt['passed'] is True
materializer=Path(c['materializer']['path']).resolve();assert c['materializer']['sha256']==sha(materializer)==receipt['materializer']['sha256'] and Path(receipt['materializer']['path']).resolve()==materializer
assert Path(pre['selection']['path']).resolve()==Path(sys.argv[2]).resolve()==Path(receipt['selection']['path']).resolve() and pre['selection']['sha256']==receipt['selection']['sha256']==sha(sys.argv[2])
assert sha(sys.argv[2])==c['selection']['sha256'] and sel['oracle_reuse']['selected_reused']==188 and sel['oracle_reuse']['selected_new']==12
for key,arg in zip(('collector','generator','auditor'),sys.argv[3:6]):
 p=Path(arg).resolve();assert Path(c[key]['path']).resolve()==p and c[key]['sha256']==sha(p)
assert Path(c['support_root']['path']).resolve()==Path(sys.argv[6]).resolve()
assert Path(c['task_config']['path']).resolve()==Path(sys.argv[7]).resolve() and c['task_config']['sha256']==sha(sys.argv[7])
assert Path(c['resize_source']['path']).resolve()==Path(sys.argv[8]).resolve() and c['resize_source']['sha256']==sha(sys.argv[8])
assert pre['oracle_root']=='/root/v478_endpoint_oracle12_seed1622_20260824' and len(pre['contexts'])==12
assert {(x['episode'],x['start']) for x in pre['contexts']}=={(x['episode'],x['start']) for x in sel['oracle_reuse']['new_contexts']}
assert pre['guards']['phase_b_temporal_collection_authorized_before_audit_pass'] is False and pre['guards']['training_authorized'] is False and pre['guards']['rl_authorized'] is False
assert sha(sys.argv[9])=='201af06ea38b74fd13f63392a2f88cacc6776f1a8193deb584ce9f443d612c3d' and sha(sys.argv[9])!=sha(sys.argv[1])
PY

root_free=$(df -PB1 /root | awk 'NR==2{print $4}');data_free=$(df -PB1 "$J" | awk 'NR==2{print $4}')
[[ "$root_free" =~ ^[0-9]+$ ]] && ((root_free>=6442450944))
[[ "$data_free" =~ ^[0-9]+$ ]] && ((data_free>=3221225472))
ACTIVE_PID='';STAGE='preflight';SUCCESS=0
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]]&&kill -0 "$ACTIVE_PID" 2>/dev/null;then kill -TERM -- "-$ACTIVE_PID" 2>/dev/null||true;for _ in $(seq 1 15);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done;kill -KILL -- "-$ACTIVE_PID" 2>/dev/null||true;wait "$ACTIVE_PID" 2>/dev/null||true;fi
 ACTIVE_PID=''
}
restore(){
 terminate_active
 bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1||true
 for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&return;sleep 1;done
 return 1
}
write_failure(){
 local rc="$1";[[ -e "$FAILREP" ]]&&return 0
 V478_STAGE="$STAGE" V478_RC="$rc" "$RLPY" - "$FAILREP" "$PRE" "$OLD_PRE" "$GENLOG" "$AUDLOG" "$ORACLE/failure_receipt.json" "$GENREP" "$AUDREP" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ev(p):p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp');obj={'format':'strict-track2-v478-endpoint-oracle12-launcher-failure-v1','passed':False,'stage':os.environ['V478_STAGE'],'exit_code':int(os.environ['V478_RC']),'preregistration_sha256':sha(sys.argv[2]),'superseded_preregistration_sha256':sha(sys.argv[3]),'superseded_preregistration_executed':False,'generator_log':ev(sys.argv[4]),'auditor_log':ev(sys.argv[5]),'oracle_failure_receipt':ev(sys.argv[6]),'generation_report':ev(sys.argv[7]),'audit_receipt':ev(sys.argv[8]),'retry_under_same_lineage_authorized':False,'phase_b_temporal_collection_authorized':False,'training_authorized':False,'s1_authorized':False,'policy_updates':0,'rl_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
on_exit(){ local rc="$1";terminate_active;restore||true;if ((SUCCESS==0));then write_failure "$rc"||true;fi; }
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}')
[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=1024))
for n in wm_v218_bridge wm_v218_gpu;do screen -S "$n" -X quit >/dev/null 2>&1||true;done
for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
! ss -ltn|grep -qE ':(8005|18084) '

STAGE='endpoint_oracle12_generation';cd "$SUPPORT"
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --collector "$COL" --support-root "$SUPPORT" --task-config "$TASK" --resize-source "$RESIZE" --output "$ORACLE" >"$GENLOG" 2>&1 & ACTIVE_PID=$!
START=$(date +%s)
while kill -0 "$ACTIVE_PID" 2>/dev/null;do sleep 2;(( $(date +%s)-START<=1800 ))||{ terminate_active;exit 124; };USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1;n++}END{if(n==0)print 0;else print s}')||{ terminate_active;exit 125; };[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=24576))||{ terminate_active;exit 125; };done
wait "$ACTIVE_PID";ACTIVE_PID='';test -s "$GENREP"

STAGE='endpoint_oracle12_audit'
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 taskset -c 0-11 "$RLPY" "$AUD" --preregistration "$PRE" --selection "$SEL" --collector "$COL" --oracle-root "$ORACLE" --generation-report "$GENREP" --support-root "$SUPPORT" --task-config "$TASK" --resize-source "$RESIZE" --output "$AUDREP" >"$AUDLOG" 2>&1 & ACTIVE_PID=$!
for _ in $(seq 1 300);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done
kill -0 "$ACTIVE_PID" 2>/dev/null&&{ terminate_active;exit 124; }
wait "$ACTIVE_PID";ACTIVE_PID='';test -s "$AUDREP"

STAGE='restore_v218';restore
"$RLPY" - "$SUCCESSREP" "$PRE" "$OLD_PRE" "$GENREP" "$AUDREP" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.load(open(sys.argv[5]));assert audit['passed'] is True and audit['authorization']['phase_b_temporal_collection_authorized'] is True
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp');obj={'format':'strict-track2-v478-endpoint-oracle12-launcher-receipt-v1','passed':True,'preregistration_sha256':sha(sys.argv[2]),'superseded_preregistration_sha256':sha(sys.argv[3]),'superseded_preregistration_executed':False,'generation_report_sha256':sha(sys.argv[4]),'audit_receipt_sha256':sha(sys.argv[5]),'phase_b_temporal_collection_authorized':True,'training_authorized':False,'s1_authorized':False,'zero_update_authorized':False,'policy_updates':0,'rl_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY

SUCCESS=1;trap - EXIT INT TERM
echo V478_ENDPOINT_ORACLE12_COMPLETE
