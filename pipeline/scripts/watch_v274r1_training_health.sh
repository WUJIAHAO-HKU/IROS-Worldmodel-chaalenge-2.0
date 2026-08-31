#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819'
RUN="$OFF/runs/$NAME"
SERVICE_REG="$OFF/run_registry/v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_20260819"

for _ in $(seq 1 1000); do
  timestamp=$(date '+%F %T %Z')
  chunks=$(grep -c 'POST /chunk_step' "$SERVICE_REG/v271_bridge.log" 2>/dev/null || true)
  videos=$(find "$RUN/video/train/seed_0" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)
  event_bytes=$(stat -c %s "$RUN"/tensorboard/events.out.tfevents.* 2>/dev/null || echo 0)
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo unavailable)
  disk_available_kb=$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
  printf '[%s] chunks=%s videos=%s event_bytes=%s gpu=%s disk_available_kb=%s\n' \
    "$timestamp" "$chunks" "$videos" "$event_bytes" "$gpu" "$disk_available_kb"

  if [[ -f "$RUN/audit/V274_RETRY1_TRAINING_ACCEPTED" ]]; then
    echo V274_RETRY1_HEALTH_WATCH_ACCEPTED
    exit 0
  fi
  if grep -aqE 'Traceback|OutOfMemory|CUDA out of memory' "$RUN/launcher.log"; then
    echo V274_RETRY1_HEALTH_WATCH_TRAINING_ERROR >&2
    exit 3
  fi
  if ! screen -ls 2>/dev/null | grep -q '[.]v274r1_v271_rl'; then
    echo V274_RETRY1_HEALTH_WATCH_SCREEN_STOPPED >&2
    exit 4
  fi
  sleep 300
done

echo V274_RETRY1_HEALTH_WATCH_TIMEOUT >&2
exit 5
