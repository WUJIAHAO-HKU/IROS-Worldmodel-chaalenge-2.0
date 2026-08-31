#!/usr/bin/env bash
# Test a time-aligned latent-action residual on the validation-proven v2 trajectory model.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

output="${WAM_WAN_OUTPUT:-artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-latent-v4-pilot}"
base="${WAM_WAN_BASE_MODEL:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
baseline="${WAM_WAN_PILOT_BASELINE:-artifacts/evaluations/track2_wan_trajectory_v2_best_preview64.json}"
candidate="${WAM_WAN_PILOT_CANDIDATE:-artifacts/evaluations/track2_wan_trajectory_latent_v4_pilot_best_preview64.json}"
comparison="${WAM_WAN_PILOT_COMPARISON:-artifacts/evaluations/track2_wan_trajectory_latent_v4_pilot_comparison.json}"
full_evaluation="${WAM_WAN_PILOT_FULL_EVALUATION:-artifacts/evaluations/track2_wan_trajectory_latent_v4_pilot_full_validation.json}"

export WAM_WAN_OUTPUT="$output"
export WAM_WAN_INIT_CHECKPOINT="${WAM_WAN_INIT_CHECKPOINT:-artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2/best/track2_wan_lora.pt}"
export WAM_WAN_TRAJECTORY_CONDITIONED=1
export WAM_WAN_TRAJECTORY_HIDDEN_DIM="${WAM_WAN_TRAJECTORY_HIDDEN_DIM:-256}"
export WAM_WAN_LATENT_ACTION_CONDITIONED=1
export WAM_WAN_LATENT_ACTION_HIDDEN_DIM="${WAM_WAN_LATENT_ACTION_HIDDEN_DIM:-256}"
export WAM_WAN_TRAJECTORY_WARMUP_STEPS="${WAM_WAN_TRAJECTORY_WARMUP_STEPS:-250}"
export WAM_WAN_FREEZE_TRAJECTORY_DURING_WARMUP=1
export WAM_WAN_HIGH_MOTION_OVERSAMPLE="${WAM_WAN_HIGH_MOTION_OVERSAMPLE:-1.0}"
export WAM_WAN_LEARNING_RATE="${WAM_WAN_LEARNING_RATE:-1e-4}"
export WAM_WAN_INHERITED_LEARNING_RATE="${WAM_WAN_INHERITED_LEARNING_RATE:-2e-5}"
export WAM_WAN_STEPS="${WAM_WAN_STEPS:-3000}"
export WAM_WAN_VALIDATION_INTERVAL="${WAM_WAN_VALIDATION_INTERVAL:-500}"
export WAM_WAN_SAVE_INTERVAL="${WAM_WAN_SAVE_INTERVAL:-500}"
export WAM_WAN_SMOKE_OUTPUT="${WAM_WAN_SMOKE_OUTPUT:-artifacts/smoke/track2-wan-trajectory-latent-v4-pilot-real}"
export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$WAM_WAN_INIT_CHECKPOINT" ]] || { echo "v2 initialization checkpoint is missing" >&2; exit 1; }

completed_steps() {
  python - "$output/training_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    print(int(json.loads(Path(sys.argv[1]).read_text()).get("step", -1)))
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    print(-1)
PY
}

evaluation_is_complete() {
  local report="$1"
  local expected_samples="$2"
  python - "$report" "$expected_samples" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
    ok = (
        data.get("format") == "track2-open-loop-eval-v1"
        and int(data.get("sample_count", 0)) == int(sys.argv[2])
        and len(data.get("model_mae_by_prediction_frame", [])) == 8
    )
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

if ! evaluation_is_complete "$baseline" 64; then
  conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend track2-wan --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2/best \
    --wan-base-model "$base" --wan-inference-steps 30 --split validation --samples 64 \
    --failure-gif-count 0 --output "$baseline"
fi

if (( $(completed_steps) < WAM_WAN_STEPS )); then
  bash pipeline/scripts/run_track2_wan_formal.sh
else
  echo "v4 training already reached step ${WAM_WAN_STEPS}; reusing checkpoint"
fi

if ! evaluation_is_complete "$candidate" 64; then
  conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
    --backend track2-wan --windows "$windows" --split-manifest "$split" \
    --checkpoint-dir "$output/best" --wan-base-model "$base" --wan-inference-steps 30 \
    --split validation --samples 64 --failure-gif-count 0 --output "$candidate"
fi

conda run --no-capture-output -n go1 python pipeline/scripts/compare_track2_evaluations.py \
  --baseline "$baseline" --candidate "$candidate" --output "$comparison"

if python - "$comparison" <<'PY'
import json
import sys
from pathlib import Path

try:
    promote = bool(json.loads(Path(sys.argv[1]).read_text()).get("promote_to_full_682_evaluation"))
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    promote = False
raise SystemExit(0 if promote else 1)
PY
then
  if ! evaluation_is_complete "$full_evaluation" 682; then
    conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
      --backend track2-wan --windows "$windows" --split-manifest "$split" \
      --checkpoint-dir "$output/best" --wan-base-model "$base" --wan-inference-steps 30 \
      --split validation --samples 682 --failure-gif-count 3 --output "$full_evaluation"
  fi
fi
