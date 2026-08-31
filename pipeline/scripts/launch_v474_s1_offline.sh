#!/usr/bin/env bash
# One-shot frozen v474 public S1 only. Never launches policy/RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";S="$ROOT/pipeline/scripts"
REG="$J/v474_v473_parent_s1_seed1617_r6_20260824";RELEASE="$J/v474_v473_median4_parent_release_seed1618_20260824";STATIC="$J/v474_v473_median4_parent_static_seed1618_20260824/static_audit_receipt.json"
PRE="$REG/preregistration.json";SEL="$REG/action_only_selection.json";SELREC="$REG/action_only_selection_receipt.json";GEN="$S/generate_v474_s1_offline.py";AUD="$S/audit_v474_s1_offline.py";WRAP="$S/v474_s1_generator_wrapper.py";MAT="$REG/materialize_v474_s1_action_only_selection.py"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt";T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
NPZ="$REG/s1_inputs.npz";GENREP="$REG/s1_generation_report.json";AUDREP="$REG/s1_audit_receipt.json"
exec 9>/var/lock/v474_s1_one_shot.lock;flock -n 9||exit 73
test -s "$PRE";test -s "$SEL";test -s "$SELREC";test ! -e "$NPZ";test ! -e "$GENREP";test ! -e "$AUDREP"
test "$(sha256sum "$PRE"|awk '{print $1}')" = 'c2a904a6ef36b5466a97fccdbe2e04185cb74356377630a4649bace00a02eee3'
test "$(sha256sum "$MAT"|awk '{print $1}')" = 'f2b531a0609a0b2fe5451334b07154ba8b8479be6866d5e2ca712601f4e6e158'
test "$(sha256sum "$SEL"|awk '{print $1}')" = 'b8e8002897b9ec0e1b386eda6d72a6bc2a09c8e322764a1c6c27b96cf902bd19'
test "$(sha256sum "$SELREC"|awk '{print $1}')" = '31a54148eace7adc31fbb5ea21efb2629600d02d1c4bf4c78f69ada8ed69fcad'
test "$(sha256sum "$GEN"|awk '{print $1}')" = '0dea600bbebb6fe41f44371952154fba52353fd5298e6ce692c060e242191400'
test "$(sha256sum "$AUD"|awk '{print $1}')" = '4a673c78a26f612d6706afb81edc1bdb62cabcaedc0712944ae9068e4c84c78f'
test "$(sha256sum "$WRAP"|awk '{print $1}')" = 'f668c6c2cd4e05aa2800521afb3ca72f9d4bcabcb1e40afe0c7b179744c79c1b'
test "$(sha256sum "$RELEASE/v474_median4_c_group_e_manifest.json"|awk '{print $1}')" = '4eecc242de272787e5aeac9df11739bf0d62cb4ba42d9224d8616a5e779ef2b9'
test "$(sha256sum "$STATIC"|awk '{print $1}')" = 'a8b6f6cf972241d7ea001fe0e15b55469dddfecfb71c4a488c62517b9686f9c2'
"$RLPY" -m py_compile "$GEN" "$AUD" "$WRAP" "$MAT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
"$RLPY" - "$PRE" "$SEL" "$SELREC" "$GEN" "$AUD" "$WRAP" "$MAT" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
pre=json.load(open(sys.argv[1]));sel=json.load(open(sys.argv[2]));rec=json.load(open(sys.argv[3]));c=pre['execution_closure']
assert pre['format']=='strict-track2-v474-v473-parent-s1-preregistration-v1' and pre['status']=='preregistered_public_s1_one_shot_authorized'
assert sel['format']=='strict-track2-v474-s1-action-only-selection-v1' and rec['format']=='strict-track2-v474-s1-action-only-selection-receipt-v1' and rec['passed'] and all(rec['checks'].values())
for key,arg in zip(('generator','auditor','s1_wrapper','materializer'),sys.argv[4:8]):
 p=Path(arg).resolve();assert Path(c[key]['path']).resolve()==p and c[key]['sha256']==sha(p)
assert c['selection']['sha256']==sha(sys.argv[2]) and c['selection_receipt']['sha256']==sha(sys.argv[3])
assert pre['run_count_exact']==1 and pre['authorization']['s1_run_authorized'] is True and pre['authorization']['zero_update_authorized_before_s1_pass'] is False and pre['authorization']['policy_update_authorized'] is False and pre['authorization']['formal_rl_authorized'] is False
assert pre['orchestration_boundary']=={'launcher_is_postregistration_orchestrator':True,'launcher_must_bind_formal_preregistration_sha256':True,'launcher_is_not_part_of_execution_closure':True}
PY

ACTIVE_PID='';START=$(date +%s)
terminate_active(){
 if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null;then kill -TERM -- "-$ACTIVE_PID" 2>/dev/null||true;for _ in $(seq 1 15);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done;kill -KILL -- "-$ACTIVE_PID" 2>/dev/null||true;wait "$ACTIVE_PID" 2>/dev/null||true;fi
 ACTIVE_PID=''
}
restore(){
 terminate_active
 bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/s1_restore_v218.log" 2>&1||true
 for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&return;sleep 1;done
 return 1
}
trap restore EXIT INT TERM
for name in wm_v218_bridge wm_v218_gpu;do screen -S "$name" -X quit >/dev/null 2>&1||true;done
for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
! ss -ltn|grep -qE ':(8005|18084) '
USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1}END{if(NR==0)print 0;else print s}')
[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=1024))
cd "$ROOT"
setsid env PYTHONPATH="$PYTHONPATH" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 taskset -c 0-11 "$RLPY" "$GEN" --preregistration "$PRE" --selection "$SEL" --selection-receipt "$SELREC" --release "$RELEASE" --static-audit "$STATIC" --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$NPZ" --report "$GENREP" --device cuda --inference-batch-size 4 --reward-batch-size 32 >"$REG/s1_generator.log" 2>&1 & ACTIVE_PID=$!
while kill -0 "$ACTIVE_PID" 2>/dev/null;do
 sleep 2;NOW=$(date +%s);((NOW-START<=3300))||{ echo timeout >&2;terminate_active;exit 124; }
 USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1}END{if(NR==0)print 0;else print s}')||{ terminate_active;exit 125; }
 [[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=28672))||{ terminate_active;exit 125; }
done
wait "$ACTIVE_PID";ACTIVE_PID=''
test -s "$NPZ";test -s "$GENREP";(( $(stat -c%s "$NPZ") <= 8589934592 ))
setsid taskset -c 0-11 "$RLPY" "$AUD" --inputs "$NPZ" --generation-report "$GENREP" --preregistration "$PRE" --selection "$SEL" --selection-receipt "$SELREC" --static-audit "$STATIC" --output "$AUDREP" >"$REG/s1_auditor.log" 2>&1 & ACTIVE_PID=$!
for _ in $(seq 1 300);do kill -0 "$ACTIVE_PID" 2>/dev/null||break;sleep 1;done
kill -0 "$ACTIVE_PID" 2>/dev/null&&{ terminate_active;exit 124; }
wait "$ACTIVE_PID";ACTIVE_PID=''
restore;trap - EXIT INT TERM
echo V474_S1_COMPLETE
