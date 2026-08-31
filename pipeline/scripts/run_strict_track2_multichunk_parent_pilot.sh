#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
OUTPUT="$JOINT/multichunk_reward_aligned_left_c4_pilot50"
PREREG="$ROOT/pipeline/config/strict_track2_multichunk_parent_preregistration.json"

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL training/evaluation\n' >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing to overwrite multichunk pilot: %s\n' "$OUTPUT" >&2
  exit 4
fi
test -s "$PREREG"
install -d "$OUTPUT/audit"
cp "$PREREG" "$OUTPUT/audit/"

export PYTHONPATH="$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF"
"$PY" "$ROOT/pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py" \
  --windows "$JOINT/onpolicy_windows_full128_stride4" \
  --split-manifest "$JOINT/onpolicy_windows_full128_stride4/split_manifest.json" \
  --init-checkpoint "$JOINT/reward_aligned_exact_fp32_probability_400/checkpoint_step_000300" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --instruction-map "$JOINT/reward_alignment/exact_instruction_map128.json" \
  --output "$OUTPUT" \
  --steps 50 \
  --checkpoint-interval 25 \
  --batch-size 1 \
  --chunks 4 \
  --chunk-stride 8 \
  --arm-filter left \
  --learning-rate 3e-7 \
  --reward-loss-weight 0.02 \
  --reward-objective probability \
  --reward-probability-scale-floor 1e-5 \
  --right-weight 1.0 \
  --success-weight 3.0 \
  --late-weight 3.0 \
  --late-start 64 \
  --seed 1616 \
  --device cuda

test -s "$OUTPUT/checkpoint_step_000050/model.pt"
printf 'MULTICHUNK_PARENT_PILOT_COMPLETE output=%s\n' "$OUTPUT"
