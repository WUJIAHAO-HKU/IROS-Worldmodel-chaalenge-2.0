#!/usr/bin/env bash
# One-shot frozen v476 public S1 only. Never launches zero-update, policy training, or RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";S="$ROOT/pipeline/scripts";RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
REG="$J/v476_v475_reward_import_s1_seed1620_20260824";RELEASE="$J/v475_v474_serial_v169_release_seed1619_20260824";STATIC="$J/v475_serial_v169_interface_s0_seed1619_20260824/static_audit_receipt.json"
PRE="$REG/preregistration.json";SEL="$REG/action_only_selection.json";SELREC="$REG/action_only_selection_receipt.json";GEN="$S/generate_v476_s1_offline.py";AUD="$S/audit_v476_s1_offline.py";WRAP="$S/v476_s1_generator_wrapper.py";MAT="$J/v474_v473_parent_s1_seed1617_r6_20260824/materialize_v474_s1_action_only_selection.py"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt";T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NPZ="$REG/s1_inputs.npz";GENREP="$REG/s1_generation_report.json";AUDREP="$REG/s1_audit_receipt.json";FAILREP="$REG/s1_failure_receipt.json";SMOKELOG="$REG/reward_import_smoke.log"
exec 9>/var/lock/v476_s1_one_shot.lock;flock -n 9||exit 73
test -s "$PRE";test -s "$SEL";test -s "$SELREC";test ! -e "$NPZ";test ! -e "$GENREP";test ! -e "$AUDREP";test ! -e "$FAILREP";test ! -e "$SMOKELOG"
test "$(sha256sum "$PRE"|awk '{print $1}')" = 'fcc1654aa2f46ac199bb94cf2f250d9e9b96245f9f1054f1d42415427b6840b5'
test "$(sha256sum "$MAT"|awk '{print $1}')" = 'f2b531a0609a0b2fe5451334b07154ba8b8479be6866d5e2ca712601f4e6e158'
test "$(sha256sum "$SEL"|awk '{print $1}')" = 'b8e8002897b9ec0e1b386eda6d72a6bc2a09c8e322764a1c6c27b96cf902bd19'
test "$(sha256sum "$SELREC"|awk '{print $1}')" = '31a54148eace7adc31fbb5ea21efb2629600d02d1c4bf4c78f69ada8ed69fcad'
test "$(sha256sum "$GEN"|awk '{print $1}')" = '7c12b2e5c16b2d5bc039d20d1ea1a95b87536dba1dea332ee5ff4c3367b27f63'
test "$(sha256sum "$AUD"|awk '{print $1}')" = '8df8ca4f4cd942e6af6b9589c06bfa97682594ec742eb09380c6ccab336c7081'
test "$(sha256sum "$WRAP"|awk '{print $1}')" = 'b9fa72dd07ec8ea7df85706954d12714a340219eb49dc35ce8b2f82bda0fc506'
test "$(sha256sum "$RELEASE/v475_serial_v169_manifest.json"|awk '{print $1}')" = '3d505839d10c3d4fbd88bcc6b0e01f2802efb8b2b3047095def2209d84d64a1d'
test "$(sha256sum "$STATIC"|awk '{print $1}')" = 'a1fda80ce1aca8ffe33ac1e4e36167df48e3318c33a3ce64a3bac0d70bb037af'
PYTHONPYCACHEPREFIX=/dev/shm/v476_s1_pycache "$RLPY" -m py_compile "$GEN" "$AUD" "$WRAP"
export PYTHONPATH="$RLINF:$ROOT/pipeline:$ROOT/pipeline/scripts" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
"$RLPY" - "$PRE" "$SEL" "$SELREC" "$GEN" "$AUD" "$WRAP" "$MAT" "$RLINF" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));sel=json.load(open(sys.argv[2]));rec=json.load(open(sys.argv[3]));c=pre['execution_closure']
assert pre['format']=='strict-track2-v476-v475-reward-import-s1-preregistration-v1' and pre['status']=='preregistered_public_s1_one_shot_authorized'
assert sel['format']=='strict-track2-v474-s1-action-only-selection-v1' and rec['format']=='strict-track2-v474-s1-action-only-selection-receipt-v1' and rec['passed'] and all(rec['checks'].values())
for key,arg in zip(('generator','auditor','s1_wrapper','materializer'),sys.argv[4:8]):
 p=Path(arg).resolve();assert Path(c[key]['path']).resolve()==p and c[key]['sha256']==sha(p)
assert c['selection']['sha256']==sha(sys.argv[2]) and c['selection_receipt']['sha256']==sha(sys.argv[3])
root=Path(sys.argv[8]).resolve();items=[]
for p in [root/'pyproject.toml',*sorted((root/'rlinf').rglob('*.py'))]:items.append([p.relative_to(root).as_posix(),sha(p)])
tree=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest();r=c['rlinf_python_source_tree']
assert len(items)==568 and Path(r['path']).resolve()==root and tree==r['sha256']=='42a9c70cb61d6f9122600df4b8663e0fb9cbd9c6d7eb136538cb9b104fd237ef'
assert pre['run_count_exact']==1 and pre['authorization']['s1_run_authorized'] is True and pre['authorization']['zero_update_authorized_before_s1_pass'] is False and pre['authorization']['policy_update_authorized'] is False and pre['authorization']['formal_rl_authorized'] is False
assert pre['orchestration_boundary']=={'launcher_is_postregistration_orchestrator':True,'launcher_must_bind_formal_preregistration_sha256':True,'launcher_is_not_part_of_execution_closure':True}
PY

ACTIVE_PID='';START=$(date +%s);STAGE='preflight';SUCCESS=0
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]]&&kill -0 "$ACTIVE_PID" 2>/dev/null;then kill -TERM -- "-$ACTIVE_PID" 2>/dev/null||true;for _ in $(seq 1 15);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done;kill -KILL -- "-$ACTIVE_PID" 2>/dev/null||true;wait "$ACTIVE_PID" 2>/dev/null||true;fi
 ACTIVE_PID=''
}
restore(){
 terminate_active
 bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/s1_restore_v218.log" 2>&1||true
 for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&return;sleep 1;done
 return 1
}
write_failure(){
 local rc="$1";[[ -e "$FAILREP" ]]&&return 0
 V476_FAIL_STAGE="$STAGE" V476_FAIL_RC="$rc" "$RLPY" - "$FAILREP" "$PRE" "$GEN" "$AUD" "$SMOKELOG" "$REG/s1_generator.log" "$NPZ" "$GENREP" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evidence(p):
 p=Path(p);return {'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.tmp')
obj={'format':'strict-track2-v476-s1-failure-receipt-v1','passed':False,'stage':os.environ['V476_FAIL_STAGE'],'exit_code':int(os.environ['V476_FAIL_RC']),'preregistration_sha256':sha(sys.argv[2]),'generator_sha256':sha(sys.argv[3]),'auditor_sha256':sha(sys.argv[4]),'reward_import_smoke_log':evidence(sys.argv[5]),'generator_log':evidence(sys.argv[6]),'s1_inputs':evidence(sys.argv[7]),'generation_report':evidence(sys.argv[8]),'zero_update_authorized':False,'endpoint_parent_data_authorized':False,'policy_updates':0,'formal_rl_authorized':False,'retry_under_v476_authorized':False}
with tmp.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p);fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
PY
}
on_exit(){ local rc="$1";terminate_active;restore||true;if ((SUCCESS==0));then write_failure "$rc"||true;fi; }
trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
STAGE='reward_import_and_t5_config_smoke'
setsid env PYTHONPATH="$PYTHONPATH" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false taskset -c 0-11 "$RLPY" - "$RLINF" "$T5" <<'PY' >"$SMOKELOG" 2>&1
import hashlib,json,sys
from pathlib import Path
import rlinf.models.embodiment.reward as reward
from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
from transformers import T5Config
root=Path(sys.argv[1]).resolve();init=Path(reward.__file__).resolve();model=init.parent/'robotwin_reward_model.py'
assert init.is_relative_to(root) and model.is_file()
assert hashlib.sha256(init.read_bytes()).hexdigest()=='8a2418f6afda50db60659619303dd385c59e14a59276d55b0829e2f3fc219754'
assert hashlib.sha256(model.read_bytes()).hexdigest()=='e2c4b123d00549e7dc136332b3c5ab8c10fce332112c990d3581f0e56dfc799e'
cfg=T5Config.from_pretrained(sys.argv[2],local_files_only=True)
print(json.dumps({'passed':True,'reward_class':RoboTwinT5CrossAttnRewardModel.__name__,'reward_module':str(init),'t5_model_type':cfg.model_type,'reward_model_constructed':False},sort_keys=True))
PY
for name in wm_v218_bridge wm_v218_gpu;do screen -S "$name" -X quit >/dev/null 2>&1||true;done
for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
! ss -ltn|grep -qE ':(8005|18084) '
USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1}END{if(NR==0)print 0;else print s}')
[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=1024))
cd "$ROOT";STAGE='interface_preflight_and_s1_generation'
setsid env PYTHONPATH="$PYTHONPATH" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --selection "$SEL" --selection-receipt "$SELREC" --release "$RELEASE" --static-audit "$STATIC" --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$NPZ" --report "$GENREP" --device cuda --inference-batch-size 4 --reward-batch-size 32 >"$REG/s1_generator.log" 2>&1 & ACTIVE_PID=$!
while kill -0 "$ACTIVE_PID" 2>/dev/null;do
 sleep 2;NOW=$(date +%s);((NOW-START<=3300))||{ echo timeout >&2;terminate_active;exit 124; }
 USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1}END{if(NR==0)print 0;else print s}')||{ terminate_active;exit 125; }
 [[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=28672))||{ terminate_active;exit 125; }
done
wait "$ACTIVE_PID";ACTIVE_PID=''
test -s "$NPZ";test -s "$GENREP";(( $(stat -c%s "$NPZ") <= 8589934592 ))
STAGE='immutable_s1_audit'
setsid taskset -c 0-11 "$RLPY" "$AUD" --inputs "$NPZ" --generation-report "$GENREP" --preregistration "$PRE" --selection "$SEL" --selection-receipt "$SELREC" --static-audit "$STATIC" --output "$AUDREP" >"$REG/s1_auditor.log" 2>&1 & ACTIVE_PID=$!
for _ in $(seq 1 300);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done
kill -0 "$ACTIVE_PID" 2>/dev/null&&{ terminate_active;exit 124; }
wait "$ACTIVE_PID";ACTIVE_PID=''
STAGE='restore_v218';restore;SUCCESS=1;trap - EXIT INT TERM
echo V476_S1_COMPLETE
