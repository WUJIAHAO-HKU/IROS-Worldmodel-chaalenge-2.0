#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
CACHE="$JOINT/post_v15_residual_cache_train_start0"
INIT="$JOINT/post_v15_residual_bounded8_reward_pilot200/checkpoint_step_000200"
OUTPUT="$JOINT/post_v15_residual_protect50_continuation800"
PREREG="$ROOT/pipeline/config/strict_track2_post_v15_residual_continuation_preregistration.json"

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL training/evaluation\n' >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing to overwrite post-V15 continuation: %s\n' "$OUTPUT" >&2
  exit 4
fi
for path in "$PREREG" "$CACHE/manifest.json" "$INIT/post_v15_residual.pt"; do
  test -s "$path"
done
install -d "$OUTPUT/audit"
cp "$PREREG" "$OUTPUT/audit/"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RLINF"
"$PY" "$ROOT/pipeline/scripts/train_strict_track2_post_v15_residual.py" \
  --cache "$CACHE" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --init-checkpoint "$INIT" --output "$OUTPUT" \
  --steps 800 --checkpoint-interval 200 --batch-size 2 \
  --learning-rate 5e-5 --reward-loss-weight 0.02 --protect-weight 50 \
  --maximum-residual-255 8 --initial-gate-probability 0.05 \
  --base-channels 16 --seed 1819 --device cuda \
  >"$OUTPUT/training.log" 2>&1
test -s "$OUTPUT/checkpoint_step_000800/post_v15_residual.pt"
printf 'POST_V15_RESIDUAL_CONTINUATION_COMPLETE output=%s\n' "$OUTPUT"
