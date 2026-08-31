#!/usr/bin/env bash
# Fixed public-development S1 gate for v439; no policy update and no simulator outcome.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v439_action_causal_projection_seed1586_20260823"
WORK='/dev/shm/v439_s1_offline_seed1586_20260823'
RELEASE="$J/v439_v169_action_causal_projection_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
SOURCE="$ROOT/artifacts/datasets/aloha-agilex_clean_50/data"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
GO1='/root/miniconda3/envs/go1/bin/python'

test -s "$REG/static_contract.json"
test -s "$RELEASE/v439_action_causal_manifest.json"
test ! -e "$REG/s1_gate.json"
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
static=json.load(open(reg/'static_contract.json'))
assert static['passed'] is True
payload={
 'format':'strict-track2-v439-s1-offline-preregistration-v1',
 'classification':'public development gate only; not independent holdout',
 'selection':{'right_dev_episodes':[6,7,18,22],'right_samples':32,'phase_samples':{'early':8,'grasp':8,'postgrasp':8,'endpoint':8},'left_contract_probes_per_episode':2,'right_g0_swap_probes':4},
 'protocol':{'recursive_chunks':4,'frames_per_chunk':8,'inference_batch_size':4,'policy_updates':0,'simulator_outcomes':False},
 'decision':{'first8_hybrid_over_v169_max':0.998,'recursive32_hybrid_over_v169_max':1.002,'reward_mae_hybrid_over_v169_max':1.0,'endpoint_reward_mae_hybrid_over_v169_max':1.0,'final_reward_hybrid_over_v169_min':0.95,'true_action_over_shuffle_target_mae_max':0.998,'right_intervention_fraction_min':0.20,'right_intervention_fraction_max':0.80,'all_required':True},
 'evidence_sha256':{name:sha(root/path) for name,path in {
  'generator':pathlib.Path('pipeline/scripts/generate_v439_s1_offline_inputs.py'),
  'auditor':pathlib.Path('pipeline/scripts/audit_v439_s1_offline.py'),
  'static_report':reg.relative_to(root)/'static_contract.json',
 }.items()},
 'guards':{'hidden_or_final_data':False,'real_submission':False,'rl_authorized':False},
}
out=reg/'s1_preregistration.json'
if out.exists():
 if json.load(open(out)) != payload: raise RuntimeError('existing S1 preregistration drifted')
else:
 json.dump(payload,open(out,'w'),indent=2)
PY

cleanup() {
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/s1_restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

for name in wm_v218_bridge wm_v218_gpu; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  ! ss -ltn | grep -qE ':(8005|18084) ' && break
  sleep 1
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline" taskset -c 0-11 "$GO1" \
  pipeline/scripts/generate_v439_s1_offline_inputs.py \
  --release "$RELEASE" --windows "$WINDOWS" --source-data "$SOURCE" \
  --split "$SPLIT" --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --output "$WORK/s1_inputs.npz" --device cuda \
  --inference-batch-size 4 --reward-batch-size 32 \
  >"$REG/s1_generator.log" 2>&1

PYTHONPATH="$ROOT/pipeline" taskset -c 0-11 "$GO1" \
  pipeline/scripts/audit_v439_s1_offline.py \
  --npz "$WORK/s1_inputs.npz" --static-report "$REG/static_contract.json" \
  --output "$REG/s1_gate.json" >"$REG/s1_auditor.log" 2>&1

echo V439_S1_COMPLETE
