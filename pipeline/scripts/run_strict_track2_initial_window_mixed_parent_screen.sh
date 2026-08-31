#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
PILOT=${TRACK2_INITIAL_WINDOW_OUTPUT:-$JOINT/initial_window_mixed_reward_aligned_lr3e7_pilot50}
SCREEN="$PILOT/screen"
BASELINE="$JOINT/formal_parent_onpolicy_expert_closedloop8_2000/checkpoints/checkpoint_step_002000"
OFFICIAL_WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
OFFICIAL_SPLIT="$ROOT/artifacts/splits/adjust_bottle_50episodes_full.json"
MIXED_WINDOWS="$JOINT/formal_joint_parent_windows_full128"

if pgrep -af '[e]val_embodied_agent.py|[t]rain_embodied_agent.py' >/dev/null; then
  printf 'Refusing to contend with active official RL training/evaluation\n' >&2
  exit 3
fi
install -d "$SCREEN"
export PYTHONPATH="$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF"

for step in 25 50; do
  candidate="$PILOT/checkpoint_step_$(printf '%06d' "$step")"
  test -s "$candidate/model.pt"
  prefix="$SCREEN/step$(printf '%03d' "$step")"

  "$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_arm_routed_autoregressive_candidate.py" \
    --windows "$OFFICIAL_WINDOWS" \
    --split-manifest "$OFFICIAL_SPLIT" \
    --episodes-key validation_episodes \
    --baseline "$BASELINE" \
    --left-candidate "$candidate" \
    --right-candidate "$candidate" \
    --min-start 0 --max-start 0 --batch-size 2 \
    --output "${prefix}_official_visual.json" \
    >"${prefix}_official_visual.log" 2>&1

  "$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_arm_routed_autoregressive_candidate.py" \
    --windows "$MIXED_WINDOWS" \
    --split-manifest "$MIXED_WINDOWS/split_manifest.json" \
    --episodes-key validation_episodes \
    --baseline "$BASELINE" \
    --left-candidate "$candidate" \
    --right-candidate "$candidate" \
    --min-start 0 --max-start 0 --batch-size 2 \
    --output "${prefix}_mixed_visual.json" \
    >"${prefix}_mixed_visual.log" 2>&1

  "$PY" "$ROOT/pipeline/scripts/export_strict_track2_arm_routed_reward_cache.py" \
    --windows "$MIXED_WINDOWS" \
    --split-manifest "$MIXED_WINDOWS/split_manifest.json" \
    --episodes-key validation_episodes \
    --baseline "$BASELINE" \
    --left-candidate "$candidate" \
    --right-candidate "$candidate" \
    --min-start 0 --max-start 0 \
    --output "${prefix}_mixed_reward_cache.npz" \
    >"${prefix}_mixed_reward_cache.log" 2>&1

  "$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_reward_alignment.py" \
    --cache "${prefix}_mixed_reward_cache.npz" \
    --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
    --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
    --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
    --output "${prefix}_mixed_reward.json" \
    --prompts-per-arm 4 --batch-size 32 --device cuda \
    >"${prefix}_mixed_reward.log" 2>&1
done

printf 'INITIAL_WINDOW_MIXED_PARENT_SCREEN_COMPLETE output=%s\n' "$SCREEN"
