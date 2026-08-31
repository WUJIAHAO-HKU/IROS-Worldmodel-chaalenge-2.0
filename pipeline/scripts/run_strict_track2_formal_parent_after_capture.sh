#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
CAPTURE_PID=${2:-0}
PY=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
GO_PY=/root/miniconda3/envs/go1/bin/python
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
BASE="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"
SYNTHETIC="$JOINT/onpolicy_windows_full128_stride4"
MIXTURE="$JOINT/formal_joint_parent_windows_full128"
EXPERT="$JOINT/formal_parent_onpolicy_expert_closedloop8_2000"
GATE="$JOINT/formal_visual_source_gate_full128"
REPORTS="$JOINT/formal_parent_expert_reports"
SELECTION="$JOINT/formal_parent_expert_selection.json"

if [[ "$CAPTURE_PID" =~ ^[1-9][0-9]*$ ]]; then
  while kill -0 "$CAPTURE_PID" 2>/dev/null; do sleep 10; done
fi
for batch in $(seq 0 7); do
  id=$(printf '%02d' "$batch")
  driver="$JOINT/onpolicy_capture/batch_${id}_driver.log"
  if [[ -s "$driver" ]]; then
    grep -Fq "CAPTURE_COMPLETE batch=$id envs=16 chunks=400" "$driver"
  elif [[ "$id" != 00 ]]; then
    printf 'Missing capture audit log: %s\n' "$driver" >&2
    exit 2
  fi
  chunks="$JOINT/onpolicy_capture/batch_${id}/chunks"
  [[ $(find "$chunks" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | wc -l) -eq 16 ]]
  [[ $(find "$chunks" -type f -name 'chunk_*.npz' | wc -l) -eq 400 ]]
  while IFS= read -r seed_dir; do
    [[ $(find "$seed_dir" -maxdepth 1 -type f -name 'chunk_*.npz' | wc -l) -eq 25 ]]
  done < <(find "$chunks" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort)
done

cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$SYNTHETIC/split_manifest.json" ]]; then
  "$PY" pipeline/scripts/convert_strict_track2_onpolicy_capture.py \
    --capture-root "$JOINT/onpolicy_capture" \
    --output "$SYNTHETIC" \
    --seed-manifest "$JOINT/onpolicy_train_seeds128/manifest.json" \
    --stride 4 --resize 256 --expected-chunks 25 \
    --validation-count 16 --split-seed 20260810
fi

if [[ ! -s "$MIXTURE/window_sources.json" ]]; then
  "$PY" pipeline/scripts/prepare_strict_track2_joint_parent_windows.py \
    --official-windows "$ROOT/artifacts/adjust_bottle_windows_full" \
    --official-split "$ROOT/artifacts/splits/adjust_bottle_38train_2dev_v19.json" \
    --synthetic-windows "$SYNTHETIC" \
    --synthetic-split "$SYNTHETIC/split_manifest.json" \
    --output "$MIXTURE" --right-repeat-factor 2
fi

if [[ ! -s "$GATE/source_gate.pt" ]]; then
  "$PY" pipeline/scripts/train_strict_track2_source_gate.py \
    --windows "$MIXTURE" --source-manifest "$MIXTURE/window_sources.json" \
    --output "$GATE" --steps 1000 --learning-rate .02 \
    --max-train-per-source 2000 --seed 20260810
fi

if [[ ! -s "$EXPERT/training_state.pt" ]] || \
   [[ $("$PY" -c 'import torch,sys; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["step"])' "$EXPERT/training_state.pt") -lt 2000 ]]; then
  RESUME=()
  if [[ -s "$EXPERT/training_state.pt" ]]; then RESUME=(--resume); fi
  "$PY" pipeline/scripts/train_autoregressive_unet.py \
    --windows "$SYNTHETIC" --split-manifest "$SYNTHETIC/split_manifest.json" \
    --output "$EXPERT" --init-autoregressive-checkpoint "$BASE" \
    --normalization-checkpoint "$BASE" --steps 2000 --batch-size 1 \
    --learning-rate 5e-6 --train-rollout-steps 8 \
    --motion-weight 3 --motion-threshold .02 --horizon-loss-power .5 \
    --temporal-delta-weight 1 --texture-laplacian-weight .5 \
    --high-motion-threshold .04 --high-motion-oversample-factor 3 \
    --right-arm-oversample-factor 1.5 \
    --high-motion-selection-weight .5 --validation-interval 200 \
    --validation-batches 64 --checkpoint-interval 200 \
    --statistics-cache "$JOINT/formal_onpolicy_expert_training_stats.npz" \
    --seed 20260810 "${RESUME[@]}"
fi

mkdir -p "$REPORTS"
for start in 200 1200; do
  pids=()
  for step in $(seq "$start" 200 $((start + 800))); do
    tag=$(printf '%06d' "$step")
    report="$REPORTS/step_${tag}_synthetic_validation.json"
    if [[ ! -s "$report" ]]; then
      "$PY" pipeline/scripts/evaluate_strict_track2_autoregressive_candidate.py \
        --windows "$SYNTHETIC" --split-manifest "$SYNTHETIC/split_manifest.json" \
        --baseline "$BASE" --candidate "$EXPERT/checkpoints/checkpoint_step_$tag" \
        --output "$report" --batch-size 1 >"${report%.json}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

"$PY" pipeline/scripts/select_strict_track2_onpolicy_expert.py \
  --reports "$REPORTS"/step_*_synthetic_validation.json --output "$SELECTION"
SELECTED=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$SELECTION")

"$PY" pipeline/scripts/evaluate_strict_track2_gated_autoregressive_candidate.py \
  --windows "$SYNTHETIC" --split-manifest "$SYNTHETIC/split_manifest.json" \
  --baseline "$BASE" --candidate "$SELECTED" --source-gate "$GATE/source_gate.pt" \
  --output "$JOINT/formal_gated_parent_synthetic_validation.json" --batch-size 1
"$PY" pipeline/scripts/evaluate_strict_track2_gated_autoregressive_candidate.py \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$ROOT/artifacts/splits/adjust_bottle_38train_2dev_v19.json" \
  --baseline "$BASE" --candidate "$SELECTED" --source-gate "$GATE/source_gate.pt" \
  --output "$JOINT/formal_gated_parent_official_dev269.json" --batch-size 1
"$PY" pipeline/scripts/screen_strict_track2_parent_candidate.py \
  --onpolicy-report "$JOINT/formal_gated_parent_synthetic_validation.json" \
  --demo-report "$JOINT/formal_gated_parent_official_dev269.json" \
  --output "$JOINT/formal_gated_parent_screen.json"

"$PY" pipeline/scripts/package_strict_track2_v15_gated_release.py \
  --base-release "$ROOT/artifacts/releases/track2_v15_best" \
  --adapted-autoregressive "$SELECTED" --source-gate "$GATE" \
  --screen-report "$JOINT/formal_gated_parent_screen.json" \
  --output "$JOINT/formal_v15_gated_onpolicy_release" \
  --model-version track2-v15.1-gated-onpolicy-formal

RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
cd "$RUNTIME"
PYTHONPATH="pipeline:$ROOT/pipeline/scripts" "$GO_PY" \
  pipeline/scripts/evaluate_strict_track2_v15_gated_composite.py \
  --windows "$SYNTHETIC" --split-manifest "$SYNTHETIC/split_manifest.json" \
  --baseline-release "$ROOT/artifacts/releases/track2_v15_best" \
  --gated-release "$JOINT/formal_v15_gated_onpolicy_release" \
  --library "$ROOT/artifacts" --output "$JOINT/formal_v15_gated_synthetic_val64.json" \
  --max-windows 64 --output-strengths 1 --device cuda
PYTHONPATH="pipeline:$ROOT/pipeline/scripts" "$GO_PY" \
  pipeline/scripts/evaluate_strict_track2_v15_gated_composite.py \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$ROOT/artifacts/splits/adjust_bottle_38train_2dev_v19.json" \
  --baseline-release "$ROOT/artifacts/releases/track2_v15_best" \
  --gated-release "$JOINT/formal_v15_gated_onpolicy_release" \
  --library "$ROOT/artifacts" --output "$JOINT/formal_v15_gated_official_dev16.json" \
  --max-windows 16 --output-strengths 1 --device cuda
"$PY" "$ROOT/pipeline/scripts/screen_strict_track2_parent_candidate.py" \
  --onpolicy-report "$JOINT/formal_v15_gated_synthetic_val64.json" \
  --demo-report "$JOINT/formal_v15_gated_official_dev16.json" \
  --output "$JOINT/formal_v15_gated_full_composite_screen.json"
printf 'FORMAL_PARENT_COMPLETE selection=%s\n' "$SELECTED"
