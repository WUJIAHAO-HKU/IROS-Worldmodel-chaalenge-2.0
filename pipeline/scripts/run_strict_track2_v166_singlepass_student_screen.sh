#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
RUNTIME="${TRACK2_REWARD_RUNTIME:-/root/autodl-tmp/iros_v15_rl_probability_audit_v2}"
PREREG="$ROOT/pipeline/config/strict_track2_v166_singlepass_student_screen_preregistration.json"
CANDIDATE="$J/v166_singlepass_student_pilot_seed166/best"
TRAINING="$CANDIDATE/training_manifest.json"
WINDOWS="$J/formal_joint_parent_windows_full128"
V157_CACHE="$J/v157_exact_reward_cache64.npz"
SCREEN="$J/v166_singlepass_student_pilot_seed166/screen"
VISUAL="$SCREEN/v166_visual_vs_v157_cache64.json"
PREDICTIONS="$SCREEN/v166_prediction_cache64.npz"
REWARD="$SCREEN/v166_official_reward_cache64.json"
RESULT="$SCREEN/v166_offline_screen.json"

for path in "$PY" "$PREREG" "$TRAINING" "$V157_CACHE"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required V16.6 screen input: $path" >&2
    exit 2
  fi
done
if [[ -e "$SCREEN" ]]; then
  echo "refusing to overwrite V16.6 screen directory: $SCREEN" >&2
  exit 3
fi
if pgrep -af 'main_grpo|MultiStepRolloutWorker|EmbodiedFSDPActor|train_strict_track2_singlepass_visual_student.py' \
  | grep -v "$$" >/dev/null; then
  echo "official RL or V16.6 training is active; refusing GPU overlap" >&2
  exit 4
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$PREREG" "$ROOT" <<'PY'
import hashlib, json, pathlib, sys
prereg_path, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
doc = json.loads(prereg_path.read_text())
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
checks = [
    (doc['pilot_preregistration']['path'], doc['pilot_preregistration']['sha256']),
    (doc['visual_screen']['script'], doc['visual_screen']['script_sha256']),
    (doc['visual_screen']['v157_cache'], doc['visual_screen']['v157_cache_sha256']),
    (doc['reward_screen']['script'], doc['reward_screen']['script_sha256']),
    (doc['reward_screen']['reward_checkpoint'], doc['reward_screen']['reward_checkpoint_sha256']),
    (doc['reward_screen']['reset_manifest'], doc['reward_screen']['reset_manifest_sha256']),
    (doc['reward_screen']['instruction_map'], doc['reward_screen']['instruction_map_sha256']),
]
for relative, expected in checks:
    path = root / relative
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f'preregistered hash mismatch: {path}: {actual} != {expected}')
print(json.dumps({'preflight_hashes': len(checks), 'passed': True}))
PY

mkdir -p "$SCREEN"
"$PY" pipeline/scripts/evaluate_strict_track2_singlepass_student_against_v157_cache.py \
  --windows "$WINDOWS" \
  --v157-cache "$V157_CACHE" \
  --v157-cache-sha256 6313ab2358714631ff610d04cc19ffa891b324fea4ada5a92732815270f90346 \
  --student "$CANDIDATE" \
  --output "$VISUAL" \
  --prediction-cache "$PREDICTIONS" \
  --benchmark-total 32 \
  --benchmark-micro-batch 8 \
  --device cuda \
  > "$SCREEN/visual.log" 2>&1

"$PY" pipeline/scripts/evaluate_strict_track2_reward_alignment.py \
  --cache "$PREDICTIONS" \
  --reuse-baseline-cache "$V157_CACHE" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --instruction-map "$J/reward_alignment/exact_instruction_map128.json" \
  --output "$REWARD" \
  --batch-size 32 \
  --device cuda \
  > "$SCREEN/reward.log" 2>&1

"$PY" pipeline/scripts/screen_strict_track2_v166_singlepass_student.py \
  --preregistration "$PREREG" \
  --training-manifest "$TRAINING" \
  --visual "$VISUAL" \
  --reward "$REWARD" \
  --candidate "$CANDIDATE" \
  --output "$RESULT" \
  | tee "$SCREEN/screen.log"
