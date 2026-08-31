#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?project root required}
J=${2:?artifact root required}
D="$J/v159_temporal_blend_sweep"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
NAMES=(endpoint12_mid100 endpoint12_shoulder50_mid100 endpoint12_ramp endpoint15_mid100 endpoint10_mid100)

for name in "${NAMES[@]}"; do
  env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$PY" \
    "$ROOT/pipeline/scripts/evaluate_strict_track2_cached_candidate_visual.py" \
    --cache "$D/$name.npz" \
    --output "$D/${name}_visual.json" \
    > "$D/${name}_visual.log" 2>&1
  env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts:$RUNTIME/pipeline:$RLINF" "$PY" \
    "$ROOT/pipeline/scripts/evaluate_strict_track2_reward_alignment.py" \
    --cache "$D/$name.npz" \
    --reuse-baseline-cache "$J/v155_action_gated_onpolicy_reward_cache64.npz" \
    --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
    --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
    --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
    --instruction-map "$J/reward_alignment/exact_instruction_map128.json" \
    --output "$D/${name}_reward.json" \
    --batch-size 32 \
    --device cuda \
    > "$D/${name}_reward.log" 2>&1
done
