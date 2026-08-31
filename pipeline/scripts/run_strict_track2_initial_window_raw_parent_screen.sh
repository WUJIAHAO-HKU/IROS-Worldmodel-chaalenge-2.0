#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
SPLIT="$ROOT/artifacts/splits/adjust_bottle_50episodes_full.json"
BASELINE="$JOINT/formal_parent_onpolicy_expert_closedloop8_2000/checkpoints/checkpoint_step_002000"
LEFT="$JOINT/reward_aligned_exact_fp32_probability_400/checkpoint_step_000300"
RIGHT="$JOINT/reward_aligned_exact_fp32_probability_400/checkpoint_step_000400"
OUTPUT="$JOINT/initial_window_same_domain_screen"
PREREG="$ROOT/pipeline/config/strict_track2_initial_window_joint_preregistration.json"

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL training/evaluation\n' >&2
  exit 3
fi
for path in "$PREREG" "$BASELINE/model.pt" "$LEFT/model.pt" "$RIGHT/model.pt"; do
  test -s "$path"
done
install -d "$OUTPUT/audit"
cp -f "$PREREG" "$OUTPUT/audit/"

export PYTHONPATH="$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF"
"$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_arm_routed_autoregressive_candidate.py" \
  --windows "$WINDOWS" \
  --split-manifest "$SPLIT" \
  --episodes-key validation_episodes \
  --baseline "$BASELINE" \
  --left-candidate "$LEFT" \
  --right-candidate "$RIGHT" \
  --min-start 0 \
  --max-start 0 \
  --batch-size 2 \
  --output "$OUTPUT/official_start0_visual.json" \
  >"$OUTPUT/official_start0_visual.log" 2>&1

"$PY" "$ROOT/pipeline/scripts/export_strict_track2_arm_routed_reward_cache.py" \
  --windows "$WINDOWS" \
  --split-manifest "$SPLIT" \
  --episodes-key validation_episodes \
  --baseline "$BASELINE" \
  --left-candidate "$LEFT" \
  --right-candidate "$RIGHT" \
  --min-start 0 \
  --max-start 0 \
  --output "$OUTPUT/official_start0_reward_cache.npz" \
  >"$OUTPUT/official_start0_reward_cache.log" 2>&1

"$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_reward_alignment.py" \
  --cache "$OUTPUT/official_start0_reward_cache.npz" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --output "$OUTPUT/official_start0_reward.json" \
  --prompts-per-arm 4 \
  --batch-size 32 \
  --device cuda \
  >"$OUTPUT/official_start0_reward.log" 2>&1

printf 'INITIAL_WINDOW_RAW_PARENT_SCREEN_COMPLETE output=%s\n' "$OUTPUT"
