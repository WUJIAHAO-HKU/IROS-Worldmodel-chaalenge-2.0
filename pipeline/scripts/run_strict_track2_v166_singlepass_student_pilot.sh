#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PYTHON_BIN="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
PREREG="$ROOT/pipeline/config/strict_track2_v166_singlepass_student_pilot_preregistration.json"
TEACHER="$J/v166_teacher_cache_actor_offload13_taskarms/manifest.json"
WINDOWS="$J/formal_joint_parent_windows_full128"
SPLIT="$WINDOWS/split_manifest.json"
INITIAL="$J/v157_hybrid_gate_blend12_formal_release/v10/flow_base"
OUTPUT="$J/v166_singlepass_student_pilot_seed166"
LOG="$J/v166_singlepass_student_pilot_seed166.log"

for required in \
  "$PYTHON_BIN" \
  "$PREREG" \
  "$TEACHER" \
  "$SPLIT" \
  "$INITIAL/model.pt" \
  "$INITIAL/action_normalization.npz" \
  "$INITIAL/track2_multisource_flow_unet_config.npz"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT" || -e "$LOG" ]]; then
  echo "refusing to overwrite V16.6 pilot output or log" >&2
  exit 3
fi

# The pilot needs backward memory.  Never overlap it with the official RLinf
# actor/rollout or another parent-training process on the single Track 2 GPU.
if pgrep -af 'main_grpo|MultiStepRolloutWorker|EmbodiedFSDPActor|train_strict_track2_singlepass_visual_student.py' \
  | grep -v "$$" >/dev/null; then
  echo "official RL or another V16.6 trainer is still active; refusing GPU overlap" >&2
  exit 4
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-25}"

exec "$PYTHON_BIN" pipeline/scripts/train_strict_track2_singlepass_visual_student.py \
  --preregistration "$PREREG" \
  --teacher-manifest "$TEACHER" \
  --ground-truth-windows "$WINDOWS" \
  --ground-truth-split "$SPLIT" \
  --initial-checkpoint "$INITIAL" \
  --output "$OUTPUT" \
  --steps 200 \
  --gradient-accumulation 4 \
  --teacher-probability 0.5 \
  --teacher-loss-scale 0.35 \
  --learning-rate 2e-5 \
  --validation-interval 50 \
  --validation-samples 32 \
  --seed 166 \
  --device cuda 2>&1 | tee "$LOG"
