#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v234_v218_public_action_reward_ranking_20260818'
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
CAPTURE="$OFF/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"

"$PY" "$ROOT/pipeline/scripts/prepare_v234_v218_action_reward_ranking.py" \
  >"$OFF/run_registry/$NAME.prepare.log" 2>&1
mv "$OFF/run_registry/$NAME.prepare.log" "$REG/prepare.log"

curl -fsS http://127.0.0.1:8005/v1/health | \
  grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'

cd /tmp
PYTHONPATH="$ROOT/pipeline:$RLINF" \
TOKENIZERS_PARALLELISM=false \
"$PY" "$ROOT/pipeline/scripts/replay_action_reward_alignment.py" \
  --audit-dir "$CAPTURE" \
  --url http://127.0.0.1:8005 \
  --token local-dev-token \
  --model-version track2-v218-public-knn-blend-alpha070-route-aware \
  --reward-checkpoint "$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$ROOT/artifacts/official_resources/reward_model/t5-base" \
  --output "$REG/action_reward_ranking.json" \
  >"$REG/action_reward_ranking.log" 2>&1

touch "$REG/AUDIT_COMPLETE"
echo V234_ACTION_REWARD_RANKING_COMPLETE
