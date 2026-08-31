#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?project root required}
J=${2:?artifact root required}
D="$J/v161_instruction_arm_reward_grid"
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY=/root/miniconda3/envs/go1/bin/python
NAMES=(alpha0 alpha0p05 alpha0p1 alpha0p12 alpha0p2 alpha0p35 alpha0p5 alpha0p75 alpha1)

for name in "${NAMES[@]}"; do
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
