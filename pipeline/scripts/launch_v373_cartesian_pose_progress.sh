#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v373_cartesian_pose_progress_seed1536_20260823"
PY=/root/miniconda3/envs/go1/bin/python

test -s "$RUN/release_registration.json"
test ! -e "$RUN/model"
if pgrep -af '[t]rain_embodied_agent.py|[e]val_embodied_agent.py|[t]rain_action_pose_v170.py' >/dev/null; then
  printf 'Refusing to contend with active training/evaluation\n' >&2
  exit 3
fi

export PYTHONPATH="$ROOT/pipeline"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
taskset -c 0-7 "$PY" "$ROOT/pipeline/scripts/train_action_pose_v170.py" \
  --dataset "$ROOT/artifacts/datasets/aloha-agilex_clean_50/data" \
  --split "$ROOT/artifacts/splits/adjust_bottle_50episodes_full.json" \
  --output "$RUN/model" --dev-episodes 36,47 \
  --steps 2200 --batch-size 512 --learning-rate 0.002 --hidden 256 \
  --seed 1536 --device cuda --log-every 100 >"$RUN/training.log" 2>&1

test -s "$RUN/model/best.pt"
test -s "$RUN/model/training_manifest.json"
touch "$RUN/RECONSTRUCTION_COMPLETE"
printf 'V373_RECONSTRUCTION_COMPLETE\n'
