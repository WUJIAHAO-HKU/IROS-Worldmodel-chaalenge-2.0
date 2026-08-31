#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v416_reward_monotone_successor_sweep_20260823'
REG="$O/run_registry/$NAME"
TRACE="$O/runs/v415_v169_v409_reward_trace32_seed1570_20260823/audit/reward_trace"
V415="$O/run_registry/v415_v169_v409_reward_trace32_seed1570_20260823/reward_signal_analysis.json"
MAP="$J/v243b_public_clean_successor_reward_map_seed1445_20260819/audit/clean_successor_rewards.npz"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
GO1='/root/miniconda3/envs/go1/bin/python'
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"

test ! -e "$REG"
mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$RLINF" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  taskset -c 0-11 "$GO1" pipeline/scripts/analyze_v416_reward_monotone_successor_sweep.py \
  --trace-dir "$TRACE" --v415-report "$V415" --reward-map "$MAP" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --device cuda --batch-size 48 \
  --output "$REG/result.json" >"$REG/analysis.log" 2>&1
test -s "$REG/result.json"
echo V416_DISCOVERY_COMPLETE
