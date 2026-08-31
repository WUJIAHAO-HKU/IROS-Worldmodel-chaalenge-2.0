#!/usr/bin/env bash
# One-shot public-development S1 gate for frozen v442 close-only candidate.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v442_close_aligned_projection_seed1593_20260823"
WORK='/dev/shm/v442_close_s1_seed1594_20260823'
RELEASE="$J/v442_v169_close_aligned_projection_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
SOURCE="$ROOT/artifacts/datasets/aloha-agilex_clean_50/data"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
GO1='/root/miniconda3/envs/go1/bin/python'

test -s "$REG/s0_audit.json"
test -s "$RELEASE/v442_close_aligned_manifest.json"
test ! -e "$REG/s1_gate.json"
test ! -e "$REG/s1_preregistration.json"
test ! -e "$WORK"
mkdir -p "$WORK"

"$GO1" - "$REG" "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
reg=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
s0=json.load(open(reg/'s0_audit.json')); assert s0['passed'] is True
coverage=s0['declared_s1_action_coverage']
files={'generator':root/'pipeline/scripts/generate_v442_s1_offline_inputs.py','generator_core':root/'pipeline/scripts/generate_v440_s1_offline_inputs.py','auditor':root/'pipeline/scripts/audit_v442_s1_offline.py','auditor_core':root/'pipeline/scripts/audit_v440_s1_offline.py','s0_report':reg/'s0_audit.json','release_manifest':root/'artifacts/strict_track2_joint_augmentation_20260810/v442_v169_close_aligned_projection_release/v442_close_aligned_manifest.json'}
payload={
 'format':'strict-track2-v442-close-s1-preregistration-v1',
 'classification':'public development gate only; not independent holdout',
 'selection':{'right_dev_episodes':[6,7,18,22],'right_samples':32,'phase_samples':{'early':8,'grasp':8,'postgrasp':8,'endpoint':8},'left_contract_probes':12,'g0_swap_probes':4},
 'protocol':{'seed':1586,'recursive_chunks':4,'frames_per_chunk':8,'inference_batch_size':4,'performance_reference':'independent recursive v169','exactness_reference':'same-request v169','policy_updates':0},
 'structural_coverage':{'active_requests_exact':8,'active_distinct_samples_exact':8,'active_phase':'grasp','active_pattern':[1,0,0,0],'other_phase_active_exact':0,'repeat_active_exact':0},
 'decision':{'first8_ratio_max':0.998,'recursive32_ratio_max':1.002,'reward_mae_ratio_max':1.0,'endpoint_reward_mae_ratio_max':1.0,'final_reward_ratio_min':0.95,'true_action_over_shuffle_mae_max':0.998,'all_required':True},
 's0_declared_coverage':coverage,'evidence_sha256':{k:sha(v) for k,v in files.items()},
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
  pipeline/scripts/generate_v442_s1_offline_inputs.py --release "$RELEASE" \
  --windows "$WINDOWS" --source-data "$SOURCE" --split "$SPLIT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$WORK/s1_inputs.npz" \
  --device cuda --inference-batch-size 4 --reward-batch-size 32 >"$REG/s1_generator.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/audit_v442_s1_offline.py --npz "$WORK/s1_inputs.npz" \
  --static-report "$REG/s0_audit.json" --output "$REG/s1_gate.json" \
  >"$REG/s1_auditor.log" 2>&1
echo V442_S1_COMPLETE
