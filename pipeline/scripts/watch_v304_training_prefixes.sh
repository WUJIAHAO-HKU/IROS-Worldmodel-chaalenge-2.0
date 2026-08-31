#!/usr/bin/env bash
set -euo pipefail

BASE="/root/autodl-tmp/IROS_WAM_2.0 challenge"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME="v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY="/root/miniconda3/envs/go1/bin/python"
LAST_STAMP=""

for _ in $(seq 1 5760); do
  event=$(ls -1t "$RUN"/tensorboard/events.out.tfevents.* 2>/dev/null | head -n 1 || true)
  if [[ -n "$event" && -f "$event" ]]; then
    mtime=$(stat -c %Y "$event")
    stamp="$event:$mtime"
    if [[ "$stamp" != "$LAST_STAMP" ]]; then
      sleep 10
      if [[ ! -f "$event" || "$(stat -c %Y "$event")" != "$mtime" ]]; then
        continue
      fi
      tmp="$RUN/audit/v304_training_prefix_latest.json.tmp"
      log_tmp="$RUN/audit/v304_training_prefix_latest.log.tmp"
      set +e
      "$PY" "$REG/audit_v304_training_prefix.py" \
        --run "$RUN" \
        --preregistration "$REG/preregistration.json" \
        --output "$tmp" > "$log_tmp" 2>&1
      rc=$?
      set -e
      if [[ -s "$tmp" ]]; then
        latest=$(
          "$PY" -c \
            'import json,sys; x=json.load(open(sys.argv[1])); print(max(x["common_train_steps"]))' \
            "$tmp"
        )
        mv "$tmp" "$RUN/audit/v304_training_prefix_step${latest}.json"
        mv "$log_tmp" "$RUN/audit/v304_training_prefix_step${latest}.log"
        cp -f "$RUN/audit/v304_training_prefix_step${latest}.json" \
          "$RUN/audit/v304_training_prefix_latest.json"
        printf 'metric_step=%s audit_rc=%s audited_at=%s\n' \
          "$latest" "$rc" "$(date -u +%FT%TZ)" \
          >> "$REG/v304_prefix_watcher_history.log"
        if (( rc != 0 )); then
          touch "$REG/V304_PREFIX_AUDIT_FAILED_STEP_${latest}"
          exit "$rc"
        fi
      else
        rm -f "$tmp"
        mv "$log_tmp" "$REG/v304_prefix_watcher_parse_failure.log"
        exit 5
      fi
      LAST_STAMP=$stamp
    fi
  fi
  if [[ -f "$RUN/audit/V304_TRAINING_ACCEPTED" ]]; then
    touch "$REG/V304_PREFIX_WATCH_COMPLETE"
    exit 0
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v304_v301_official_rl'; then
    echo V304_STOPPED_WITHOUT_TRAINING_ACCEPTANCE >&2
    exit 3
  fi
  sleep 60
done

echo V304_PREFIX_WATCH_TIMEOUT >&2
exit 4
