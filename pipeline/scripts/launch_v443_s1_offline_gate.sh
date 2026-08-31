#!/usr/bin/env bash
# One-shot public-development S1 gate for frozen v443 spatial close candidate.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v443_close_spatial_projection_seed1595_20260823"
WORK='/dev/shm/v443_close_spatial_s1_seed1596_20260823'
RELEASE="$J/v443_v169_close_spatial_projection_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
SOURCE="$ROOT/artifacts/datasets/aloha-agilex_clean_50/data"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
GO1='/root/miniconda3/envs/go1/bin/python'

test -s "$REG/s0_audit.json"
test -s "$RELEASE/v443_close_spatial_manifest.json"
test ! -e "$REG/s1_gate.json"
test ! -e "$REG/s1_preregistration.json"
test ! -e "$WORK"
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v443_s1_offline_inputs.py" | awk '{print $1}')" = \
  '940538f35d3b58860c8737bf98e30cfbe0531e94a60039beece649172f1891d3'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v443_s1_offline.py" | awk '{print $1}')" = \
  '5a6fecc97b28996babfe97246fca19450cdda89476e5c44edc6f0d82c09d3cf2'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v440_s1_offline_inputs.py" | awk '{print $1}')" = \
  'cdb18bf117c766d8b3f9e7a09900ec13a5d1013068cbda69bdab3719cb830618'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v440_s1_offline.py" | awk '{print $1}')" = \
  'b84c30ad510fdfd49a87bb263211791c469123c98c1d760f1628ec416817b540'
mkdir -p "$WORK"

"$GO1" - "$REG" "$ROOT" "$RELEASE" <<'PY'
import hashlib,json,pathlib,sys
reg=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2]); release=pathlib.Path(sys.argv[3])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
s0=json.load(open(reg/'s0_audit.json')); assert s0['passed'] is True
manifest=json.load(open(release/'v443_close_spatial_manifest.json'))
coverage=manifest['s1_action_coverage_contract']
files={'generator':root/'pipeline/scripts/generate_v443_s1_offline_inputs.py','generator_core':root/'pipeline/scripts/generate_v440_s1_offline_inputs.py','auditor':root/'pipeline/scripts/audit_v443_s1_offline.py','auditor_core':root/'pipeline/scripts/audit_v440_s1_offline.py','structure_core':root/'pipeline/scripts/audit_v442_s1_offline.py','s0_report':reg/'s0_audit.json','release_manifest':release/'v443_close_spatial_manifest.json'}
payload={
 'format':'strict-track2-v443-close-s1-preregistration-v1',
 'classification':'public development gate only; not independent holdout',
 'selection':{'right_dev_episodes':[6,7,18,22],'right_samples':32,'phase_samples':{'early':8,'grasp':8,'postgrasp':8,'endpoint':8},'left_contract_probes':12,'g0_swap_probes':4},
 'protocol':{'seed':1586,'recursive_chunks':4,'frames_per_chunk':8,'inference_batch_size':4,'performance_reference':'independent recursive v169','exactness_reference':'same-request v169','policy_updates':0},
 'structural_coverage':{'active_requests_exact':8,'active_distinct_samples_exact':8,'active_phase':'grasp','active_pattern':[1,0,0,0],'other_phase_active_exact':0,'repeat_active_exact':0},
 'decision':{'first8_ratio_max':0.998,'recursive32_ratio_max':1.002,'reward_mae_ratio_max':1.0,'endpoint_reward_mae_ratio_max':1.0,'final_reward_ratio_min':0.95,'true_action_over_shuffle_mae_max':0.998,'all_required':True},
 'release_declared_coverage':coverage,'evidence_sha256':{k:sha(v) for k,v in files.items()},
 'guards':{'coverage_contract_selected_from_actions_only':True,'performance_thresholds_changed':False,'hidden_or_final_data':False,'real_submission':False,'rl_authorized':False},
}
json.dump(payload,open(reg/'s1_preregistration.json','w'),indent=2)
PY

cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/s1_restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/generate_v443_s1_offline_inputs.py --release "$RELEASE" \
  --windows "$WINDOWS" --source-data "$SOURCE" --split "$SPLIT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$WORK/s1_inputs.npz" \
  --device cuda --inference-batch-size 4 --reward-batch-size 32 >"$REG/s1_generator.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/audit_v443_s1_offline.py --npz "$WORK/s1_inputs.npz" \
  --static-report "$REG/s0_audit.json" --output "$REG/s1_gate.json" \
  >"$REG/s1_auditor.log" 2>&1
echo V443_S1_COMPLETE
