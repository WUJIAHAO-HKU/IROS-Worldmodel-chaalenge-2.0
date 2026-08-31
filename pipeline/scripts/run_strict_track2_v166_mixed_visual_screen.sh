#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
PREREG="$ROOT/pipeline/config/strict_track2_v166_mixed_visual_screen_preregistration.json"
CACHE="$J/v157_balanced_mixed_visual_cache64.npz"
CACHE_MANIFEST="$J/v157_balanced_mixed_visual_cache64.manifest.json"
CANDIDATE="$J/v166_singlepass_student_pilot_seed166/best"
SCREEN_DIR="$J/v166_singlepass_student_pilot_seed166/screen"
VISUAL="$SCREEN_DIR/v166_mixed_visual_vs_v157_cache64.json"
RESULT="$SCREEN_DIR/v166_mixed_visual_screen.json"

for path in "$PREREG" "$CACHE" "$CACHE_MANIFEST" "$CANDIDATE/training_manifest.json"; do
  test -e "$path" || { echo "missing mixed-screen input: $path" >&2; exit 2; }
done
if [[ -e "$VISUAL" || -e "$RESULT" ]]; then
  echo "refusing to overwrite mixed visual screen" >&2
  exit 3
fi
cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts${PYTHONPATH:+:$PYTHONPATH}"
cache_sha=$("$PY" - "$PREREG" "$CACHE_MANIFEST" "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
p=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); root=pathlib.Path(sys.argv[3])
assert p['format']=='strict-track2-v166-mixed-visual-screen-preregistration-v1'
for block in ('cache_builder_preregistration','evaluator'):
    path=root/p[block]['path' if block=='cache_builder_preregistration' else 'script']
    expected=p[block]['sha256' if block=='cache_builder_preregistration' else 'script_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
cache=pathlib.Path(m['cache']); assert hashlib.sha256(cache.read_bytes()).hexdigest()==m['cache_sha256']
print(m['cache_sha256'])
PY
)

"$PY" pipeline/scripts/evaluate_strict_track2_singlepass_student_against_v157_cache.py \
  --windows "$J/formal_joint_parent_windows_full128" \
  --v157-cache "$CACHE" \
  --v157-cache-sha256 "$cache_sha" \
  --student "$CANDIDATE" \
  --output "$VISUAL" \
  --benchmark-total 32 \
  --benchmark-micro-batch 8 \
  --device cuda \
  > "$SCREEN_DIR/mixed_visual.log" 2>&1

"$PY" pipeline/scripts/screen_strict_track2_v166_mixed_visual.py \
  --preregistration "$PREREG" \
  --cache-manifest "$CACHE_MANIFEST" \
  --visual "$VISUAL" \
  --candidate "$CANDIDATE" \
  --output "$RESULT" \
  | tee "$SCREEN_DIR/mixed_visual_screen.log"
