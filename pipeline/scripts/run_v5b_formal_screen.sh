#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
O="${SCREEN_OUTPUT:-$J/recursive_arm_routed_v5b_right_base/formal_screen}"
PY=/root/miniconda3/envs/go1/bin/python
B="$ROOT/artifacts/releases/track2_v15_best/v8/baseline/autoregressive"
L="${LEFT_CANDIDATE:-$J/recursive_multichunk_v4b_interp_v1/alpha_0p2500}"
W="$J/onpolicy_windows_full128_stride4"
S="$W/split_manifest.json"
mkdir -p "$O"
export PYTHONPATH="$ROOT/pipeline/scripts:/root/autodl-tmp/iros_v15_rl_probability_audit_v2/pipeline"
"$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_arm_routed_autoregressive_candidate.py" \
  --windows "$W" --split-manifest "$S" --baseline "$B" \
  --left-candidate "$L" --right-candidate "$B" --batch-size 2 \
  --output "$O/visual_all_windows.json" >"$O/visual.log" 2>&1
"$PY" "$ROOT/pipeline/scripts/export_strict_track2_arm_routed_reward_cache.py" \
  --windows "$W" --split-manifest "$S" --baseline "$B" \
  --left-candidate "$L" --right-candidate "$B" --max-windows 64 \
  --output "$O/reward_cache64.npz" >"$O/cache.log" 2>&1
"$PY" "$ROOT/pipeline/scripts/evaluate_strict_track2_reward_alignment.py" \
  --cache "$O/reward_cache64.npz" \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --reset-manifest "$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json" \
  --instruction-map "$J/reward_alignment/exact_instruction_map128.json" \
  --output "$O/reward_alignment64.json" --prompts-per-arm 4 --batch-size 32 --device cuda \
  >"$O/reward.log" 2>&1
date -Is >"$O/DONE"
