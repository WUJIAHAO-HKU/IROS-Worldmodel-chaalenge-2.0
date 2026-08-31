#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_RL_PYTHON:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
DIAGNOSTIC_RUN_ID="${TRACK2_DIAGNOSTIC_RUN_ID:-probability_consistent_one_update_v157_seed1243}"
RUN="$AUDIT/runs/$DIAGNOSTIC_RUN_ID"
EXPERIMENT="wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
CHECKPOINT="$RUN/$EXPERIMENT/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
VARIANT="${TRACK2_CANDIDATE_VARIANT:-v157_probability_consistent_seed1243_historical_dev22}"
DEV_ROOT="$AUDIT/real_robotwin_eval/development_seeds22"
OUTPUT_ROOT="$AUDIT/real_robotwin_eval/development_metrics"
STATE="$RUN/audit/automatic_continuation.log"

while pgrep -f "train_embodied_agent.py.*${DIAGNOSTIC_RUN_ID}" >/dev/null; do
  printf '%s waiting_for_one_update\n' "$(date -Iseconds)" >> "$STATE"
  sleep 60
done

test -s "$CHECKPOINT"
grep -aq 'Global Step:    1/1' "$RUN/launcher.log"
if grep -aqE 'CUDA out of memory|OutOfMemoryError|HTTP.*(500|502|503|504)' "$RUN/launcher.log"; then
  printf '%s one_update_failed_runtime_gate\n' "$(date -Iseconds)" >> "$STATE"
  exit 7
fi
printf '%s starting_historical_dev22_batch00\n' "$(date -Iseconds)" >> "$STATE"

TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_SEED_ROOT="$DEV_ROOT" TRACK2_DEV_OUTPUT_ROOT="$OUTPUT_ROOT" TRACK2_DEV_BATCH_FILTER="00" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"

set +e
PYTHONPATH="$ROOT/pipeline/scripts" "$PY" "$ROOT/pipeline/scripts/screen_strict_track2_dev22_batch00.py" \
  --output-root "$OUTPUT_ROOT" --dev-root "$DEV_ROOT" --candidate "$VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_batch00_screen_$VARIANT.json"
screen_status=$?
set -e
if [[ "$screen_status" -ne 0 ]]; then
  printf '%s rejected_after_batch00 status=%s\n' "$(date -Iseconds)" "$screen_status" >> "$STATE"
  exit "$screen_status"
fi

TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_SEED_ROOT="$DEV_ROOT" TRACK2_DEV_OUTPUT_ROOT="$OUTPUT_ROOT" TRACK2_DEV_BATCH_FILTER="01" \
  "$AUDIT/real_robotwin_eval/run_strict_track2_dev_eval.sh"

PYTHONPATH="$ROOT/pipeline/scripts" "$PY" "$ROOT/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT_ROOT" --dev-root "$DEV_ROOT" --candidate "$VARIANT" \
  --output "$AUDIT/real_robotwin_eval/dev_ranking_$VARIANT.json"
printf '%s historical_dev22_complete\n' "$(date -Iseconds)" >> "$STATE"
