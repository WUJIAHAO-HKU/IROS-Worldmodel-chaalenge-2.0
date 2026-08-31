#!/usr/bin/env bash
# Fine-tune Direct Flow, compare on identical pilot windows, then optionally run full validation.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
conda_bin="${CONDA_BIN:-/root/miniconda3/bin/conda}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/go1/bin/python}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
flows="${WAM_FLOW_TARGETS:-artifacts/flow_targets/raft_small_train_v2_128}"
initialization="${WAM_DIRECT_V3_INIT:-artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best}"
output="${WAM_DIRECT_V3_OUTPUT:-artifacts/checkpoints/direct-flow-unet-track2-long-horizon-v3-pilot}"
steps="${WAM_DIRECT_V3_STEPS:-1000}"
samples="${WAM_DIRECT_V3_SAMPLES:-64}"
baseline="${WAM_DIRECT_V3_BASELINE:-artifacts/evaluations/direct_flow_formal_v2_preview64.json}"
candidate="${WAM_DIRECT_V3_CANDIDATE:-artifacts/evaluations/direct_flow_long_horizon_v3_preview64.json}"
comparison="${WAM_DIRECT_V3_COMPARISON:-artifacts/evaluations/direct_flow_long_horizon_v3_comparison.json}"
full="${WAM_DIRECT_V3_FULL:-artifacts/evaluations/direct_flow_long_horizon_v3_validation_all.json}"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/finetune_direct_flow_unet.py \
  --windows "$windows" --split-manifest "$split" --flow-targets "$flows" \
  --init-checkpoint "$initialization" --output "$output" --steps "$steps" \
  --validation-interval 250 --validation-batches "$samples" --checkpoint-interval 25 \
  --batch-size 1 --base-channels 80 --learning-rate 1e-5 --motion-weight 5 \
  --high-motion-oversample-factor 5 --high-motion-selection-weight 0.5 \
  --horizon-loss-power 0.5 --device cuda --resume

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend direct-flow-unet --windows "$windows" --split-manifest "$split" \
  --checkpoint-dir "$output/best" --split validation --samples "$samples" \
  --failure-gif-count 0 --output "$candidate"

"$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --minimum-relative-improvement 0.01 --output "$comparison"

if "$python_bin" - "$comparison" <<'PY'
import json, sys
from pathlib import Path
try:
    promote = bool(json.loads(Path(sys.argv[1]).read_text())["promote_to_full_682_evaluation"])
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
    promote = False
raise SystemExit(0 if promote else 1)
PY
then
  "$conda_bin" run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend direct-flow-unet --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$output/best" --split validation --samples 682 \
    --accept-mae 1.0 --failure-gif-count 3 --output "$full"
fi
