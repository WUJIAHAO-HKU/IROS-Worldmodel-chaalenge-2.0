#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
WINDOWS="$JOINT/formal_joint_parent_windows_full128"
BASE="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"
OUTPUT="$JOINT/v155_joint_domain_parent_3000"
PREREG="$ROOT/pipeline/config/strict_track2_joint_domain_parent_v155_preregistration.json"

test -s "$WINDOWS/split_manifest.json"
test -s "$WINDOWS/window_sources.json"
test -s "$BASE/model.pt"
test -s "$PREREG"

mkdir -p "$OUTPUT/audit"
cp "$PREREG" "$OUTPUT/audit/"
sha256sum "$PREREG" "$WINDOWS/split_manifest.json" "$WINDOWS/window_sources.json" \
  "$BASE/model.pt" >"$OUTPUT/audit/frozen_inputs.sha256"

if [[ -s "$OUTPUT/training_state.pt" ]]; then
  current_step=$(
    "$PYTHON_BIN" -c \
      'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["step"])' \
      "$OUTPUT/training_state.pt"
  )
else
  current_step=0
fi
if (( current_step >= 3000 )); then
  printf 'SKIP reason=complete step=%s output=%s\n' "$current_step" "$OUTPUT"
  exit 0
fi

resume=()
if (( current_step > 0 )); then resume=(--resume); fi
cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON_BIN" pipeline/scripts/train_autoregressive_unet.py \
  --windows "$WINDOWS" \
  --split-manifest "$WINDOWS/split_manifest.json" \
  --output "$OUTPUT" \
  --init-autoregressive-checkpoint "$BASE" \
  --normalization-checkpoint "$BASE" \
  --steps 3000 \
  --batch-size 1 \
  --learning-rate 2e-6 \
  --train-rollout-steps 8 \
  --motion-weight 3 \
  --motion-threshold .02 \
  --horizon-loss-power .5 \
  --temporal-delta-weight 1 \
  --texture-laplacian-weight .5 \
  --high-motion-threshold .04 \
  --high-motion-oversample-factor 3 \
  --high-motion-selection-weight .5 \
  --validation-interval 250 \
  --validation-batches 64 \
  --checkpoint-interval 250 \
  --statistics-cache "$JOINT/v155_joint_domain_parent_training_stats.npz" \
  --seed 20260811 \
  "${resume[@]}"
