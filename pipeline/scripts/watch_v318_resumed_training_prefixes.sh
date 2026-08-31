#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY='/root/miniconda3/envs/go1/bin/python'
AUDITOR="$BASE/pipeline/scripts/audit_v308_training_prefix.py"
LAST=''

for _ in $(seq 1 5760); do
  event=$(ls -1t "$RUN"/tensorboard/events.out.tfevents.* 2>/dev/null | head -n 1 || true)
  if [[ -n "$event" && -f "$event" ]]; then
    stamp="$event:$(stat -c %Y "$event")"
    if [[ "$stamp" != "$LAST" ]]; then
      sleep 10
      tmp="$RUN/audit/v318_resumed_training_prefix_latest.json.tmp"
      log="$RUN/audit/v318_resumed_training_prefix_latest.log.tmp"
      set +e
      "$PY" "$AUDITOR" --run "$RUN" --preregistration "$REG/preregistration.json" --output "$tmp" >"$log" 2>&1
      rc=$?
      set -e
      if [[ -s "$tmp" ]]; then
        latest=$($PY -c 'import json,sys; x=json.load(open(sys.argv[1])); s=x["common_train_steps"]; print(max(s) if s else "")' "$tmp")
        if [[ -n "$latest" ]]; then
          mv "$tmp" "$RUN/audit/v318_training_prefix_step${latest}.json"
          mv "$log" "$RUN/audit/v318_training_prefix_step${latest}.log"
          cp -f "$RUN/audit/v318_training_prefix_step${latest}.json" "$RUN/audit/v318_training_prefix_latest.json"
          printf 'metric_step=%s audit_rc=%s audited_at=%s source=resumed\n' "$latest" "$rc" "$(date -u +%FT%TZ)" >>"$REG/v318_prefix_watcher_history.log"
          if (( rc != 0 )); then touch "$REG/V318_PREFIX_AUDIT_FAILED_STEP_${latest}"; exit "$rc"; fi
        else
          rm -f "$tmp" "$log"
        fi
      fi
      LAST="$stamp"
    fi
  fi
  if [[ -f "$RUN/audit/V318_TRAINING_ACCEPTED" ]]; then touch "$REG/V318_RESUMED_PREFIX_WATCH_COMPLETE"; exit 0; fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v318_resume_from_step3'; then
    echo V318_RESUME_STOPPED_WITHOUT_ACCEPTANCE >&2
    exit 3
  fi
  sleep 60
done

exit 4
