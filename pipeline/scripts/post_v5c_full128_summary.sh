#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
EVAL="$ROOT/artifacts/strict_track2_official_20260810/real_robotwin_eval"
METRICS="$EVAL/instrumented_metrics_v5c_smoke_step1"
WATCH_LOG="$EVAL/v5c_smoke_eval_watch.log"
OUTPUT="$EVAL/v5c_smoke_step1_full128_summary.json"
POST_LOG="$EVAL/v5c_smoke_step1_full128_post.log"

while ! grep -q 'real_eval_complete=' "$WATCH_LOG" 2>/dev/null; do
  sleep 30
done

python3 "$ROOT/pipeline/scripts/summarize_strict_track2_full128.py" \
  --eval-root "$METRICS" \
  --candidate v5c_smoke_step1 \
  --reference-summary "$EVAL/v169_four_step_retry2_summary.json" \
  --output "$OUTPUT" >"$POST_LOG" 2>&1
printf '%s summary_complete=%s\n' "$(date -Iseconds)" "$OUTPUT" >> "$POST_LOG"
