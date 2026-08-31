#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
TRAIN="$JOINT/v155_joint_domain_parent_3000"
REPORTS="$JOINT/v155_joint_domain_parent_reports"
BASE="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"
PREREG="$ROOT/pipeline/config/strict_track2_joint_domain_parent_v155_preregistration.json"
SELECTION="$JOINT/v155_joint_domain_parent_selection.json"

test -s "$TRAIN/training_state.pt"
test "$($PYTHON_BIN -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["step"])' "$TRAIN/training_state.pt")" -eq 3000
mkdir -p "$REPORTS"
cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"

evaluate_domain() {
  local domain=$1 windows=$2 split=$3
  local pids=() step tag output
  for step in 500 1000 1500 2000 2500 3000; do
    tag=$(printf '%06d' "$step")
    output="$REPORTS/step_${tag}_${domain}.json"
    if [[ ! -s "$output" ]]; then
      "$PYTHON_BIN" pipeline/scripts/evaluate_strict_track2_autoregressive_candidate.py \
        --windows "$windows" --split-manifest "$split" \
        --baseline "$BASE" --candidate "$TRAIN/checkpoints/checkpoint_step_$tag" \
        --output "$output" --batch-size 1 >"${output%.json}.log" 2>&1 &
      pids+=("$!")
    fi
    # Bound each wave to three evaluators; this stays well below the 32 GB GPU.
    if [[ ${#pids[@]} -eq 3 ]]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

evaluate_domain \
  official \
  "$ROOT/artifacts/adjust_bottle_windows_full" \
  "$ROOT/artifacts/splits/adjust_bottle_38train_2dev_v19.json"
evaluate_domain \
  onpolicy \
  "$JOINT/onpolicy_windows_full128_stride4" \
  "$JOINT/onpolicy_windows_full128_stride4/split_manifest.json"

selector_args=()
for step in 500 1000 1500 2000 2500 3000; do
  tag=$(printf '%06d' "$step")
  selector_args+=(
    --official-report "$step:$REPORTS/step_${tag}_official.json"
    --onpolicy-report "$step:$REPORTS/step_${tag}_onpolicy.json"
  )
done
"$PYTHON_BIN" pipeline/scripts/select_strict_track2_joint_domain_parent.py \
  "${selector_args[@]}" \
  --checkpoint-root "$TRAIN" \
  --preregistration "$PREREG" \
  --output "$SELECTION"
printf 'JOINT_DOMAIN_PARENT_EVALUATION_COMPLETE selection=%s\n' "$SELECTION"
