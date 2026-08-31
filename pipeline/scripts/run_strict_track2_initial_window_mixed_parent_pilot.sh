#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
OUTPUT=${TRACK2_INITIAL_WINDOW_OUTPUT:-$JOINT/initial_window_mixed_reward_aligned_lr3e7_pilot50}
PREREG="$ROOT/pipeline/config/strict_track2_initial_window_joint_preregistration.json"
INIT="$JOINT/formal_parent_onpolicy_expert_closedloop8_2000/checkpoints/checkpoint_step_002000"
WINDOWS="$JOINT/formal_joint_parent_windows_full128"

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL training/evaluation\n' >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing to overwrite initial-window pilot: %s\n' "$OUTPUT" >&2
  exit 4
fi
for path in "$PREREG" "$INIT/model.pt" "$WINDOWS/split_manifest.json"; do
  test -s "$path"
done
install -d "$OUTPUT/audit"
cp "$PREREG" "$OUTPUT/audit/"

export PYTHONPATH="$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF"
"$PY" "$ROOT/pipeline/scripts/train_reward_aligned_autoregressive_unet.py" \
  --windows "$WINDOWS" \
  --split-manifest "$WINDOWS/split_manifest.json" \
  --init-checkpoint "$INIT" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --instruction-map "$JOINT/reward_alignment/exact_instruction_map128.json" \
  --output "$OUTPUT" \
  --steps 50 \
  --checkpoint-interval 25 \
  --batch-size 1 \
  --learning-rate 3e-7 \
  --reward-loss-weight 0.02 \
  --reward-objective probability \
  --reward-probability-scale-floor 1e-5 \
  --right-weight 0.73 \
  --success-weight 1.0 \
  --late-weight 1.0 \
  --late-start 80 \
  --min-start 0 \
  --max-start 0 \
  --official-weight 3.0 \
  --prompts-per-arm 4 \
  --seed 1717 \
  --device cuda \
  >"$OUTPUT/training.log" 2>&1

test -s "$OUTPUT/checkpoint_step_000050/model.pt"
printf 'INITIAL_WINDOW_MIXED_PARENT_PILOT_COMPLETE output=%s\n' "$OUTPUT"
