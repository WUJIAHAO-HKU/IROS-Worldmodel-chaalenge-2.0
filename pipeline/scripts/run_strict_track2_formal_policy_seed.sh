#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
ACTOR_SEED=${2:?actor seed is required}
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
SCREEN=${TRACK2_PARENT_SCREEN:-$JOINT/formal_v15_gated_full_composite_screen.json}
VARIANT=${TRACK2_POLICY_VARIANT:-gated_formal_seed${ACTOR_SEED}}
CHECKPOINT="$AUDIT/runs/official_replica_seed_${ACTOR_SEED}/wan_robotwin_adjust_bottle_http_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

/root/autodl-tmp/conda_envs/rlinf_track2/bin/python - <<'PY' "$SCREEN"
import json, sys
report = json.load(open(sys.argv[1]))
assert report["passed"] is True, report
PY
curl -fsS http://127.0.0.1:8001/v1/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'

TRACK2_PARENT_SCREEN="$SCREEN" \
  "$AUDIT/run_strict_track2_official_replica.sh" "$ACTOR_SEED"
test -s "$CHECKPOINT"
TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"
/root/autodl-tmp/conda_envs/rlinf_track2/bin/python \
  "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --eval-root "$AUDIT/real_robotwin_eval/development_metrics" \
  --candidate-variant "$VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_ranking_${VARIANT}.json"
printf 'FORMAL_POLICY_SEED_DEVELOPMENT_COMPLETE seed=%s checkpoint=%s\n' \
  "$ACTOR_SEED" "$CHECKPOINT"
