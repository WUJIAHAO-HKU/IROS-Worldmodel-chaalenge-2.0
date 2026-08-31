#!/usr/bin/env bash
# Train a resumable recursive Flow pilot and gate it on paired held-out windows.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
warm_start="${WAM_RECURSIVE_WARM_START:-artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best}"
output="${WAM_RECURSIVE_OUTPUT:-artifacts/checkpoints/recursive-flow-unet-track2-pilot-v1}"
steps="${WAM_RECURSIVE_STEPS:-500}"
samples="${WAM_RECURSIVE_PILOT_SAMPLES:-64}"
baseline="${WAM_RECURSIVE_BASELINE:-artifacts/evaluations/direct_flow_formal_v2_preview64.json}"
candidate="${WAM_RECURSIVE_CANDIDATE:-artifacts/evaluations/recursive_flow_pilot_v1_preview64.json}"
comparison="${WAM_RECURSIVE_COMPARISON:-artifacts/evaluations/recursive_flow_pilot_v1_comparison.json}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"

[[ -f "$warm_start/model.pt" ]] || { echo "Direct Flow warm start is missing: $warm_start" >&2; exit 1; }
[[ "$steps" =~ ^[1-9][0-9]*$ && "$samples" =~ ^[1-9][0-9]*$ ]] || {
  echo "WAM_RECURSIVE_STEPS and WAM_RECURSIVE_PILOT_SAMPLES must be positive integers" >&2
  exit 2
}

evaluation_is_complete() {
  python - "$1" "$samples" <<'PY'
import json
import sys
from pathlib import Path
try:
    report = json.loads(Path(sys.argv[1]).read_text())
    ok = report.get("format") == "track2-open-loop-eval-v1" and int(report.get("sample_count", 0)) == int(sys.argv[2])
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

if ! evaluation_is_complete "$baseline"; then
  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend direct-flow-unet --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$warm_start" --split validation --samples "$samples" \
    --failure-gif-count 0 --output "$baseline"
fi

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/train_recursive_flow_unet.py \
  --windows "$windows" --split-manifest "$split" --warm-start "$warm_start" \
  --output "$output" --steps "$steps" --base-channels 80 --learning-rate 2e-5 \
  --validation-interval 100 --validation-batches "$samples" --checkpoint-interval 25 \
  --batch-size 1 --device cuda --resume

if ! evaluation_is_complete "$candidate"; then
  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend recursive-flow-unet --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$output/best" --split validation --samples "$samples" \
    --failure-gif-count 3 --output "$candidate"
fi

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --output "$comparison"
