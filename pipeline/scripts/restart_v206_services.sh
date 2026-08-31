#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
V206="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v206_v202_v205_public_arm_routed_release"
MODEL_VERSION='track2-v206-public-arm-routed-v202-left-v205-right-step120'
TOKEN='local-dev-token'
SERVICE_URL='http://127.0.0.1:8004'
BRIDGE_URL='http://127.0.0.1:18083'

run_service() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" \
    WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="$TOKEN" \
    WAM_BACKEND='arm-routed-autoregressive-unet' WAM_CHECKPOINT_DIR="$V206" \
    WAM_DEVICE='cuda' WAM_PORT=8004 WAM_NATIVE_BATCH_ENABLED=1 \
    WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=1 \
    "$GO1" pipeline/scripts/serve.py >> "$V206/v206_service_gpu.log" 2>&1
}

run_bridge() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server \
    --world-model-url "$SERVICE_URL" --token "$TOKEN" \
    --model-version "$MODEL_VERSION" --host 127.0.0.1 --port 18083 \
    --audit-max-items 1 >> "$V206/v206_bridge.log" 2>&1
}

wait_for() {
  local url="$1" attempts="$2"
  for _ in $(seq 1 "$attempts"); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

case "${1:-start}" in
  service) run_service ;;
  bridge) run_bridge ;;
  start)
    for name in wm_v202_bridge wm_v202_gpu wm_v206_bridge wm_v206_gpu; do
      screen -S "$name" -X quit >/dev/null 2>&1 || true
    done
    for _ in $(seq 1 30); do
      if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
      sleep 1
    done
    if ss -ltn | grep -qE ':(8004|18083) '; then
      echo 'world-model ports did not stop cleanly' >&2
      exit 3
    fi
    screen -dmS wm_v206_gpu bash "$0" service
    wait_for "$SERVICE_URL/v1/health" 120
    curl -fsS "$SERVICE_URL/v1/health" | grep -q "$MODEL_VERSION"
    curl -fsS -H "Authorization: Bearer $TOKEN" "$SERVICE_URL/v1/capabilities" | grep -q "$MODEL_VERSION"
    screen -dmS wm_v206_bridge bash "$0" bridge
    wait_for "$BRIDGE_URL/health" 60
    curl -fsS "$BRIDGE_URL/health" | grep -q '"status":"ready"'
    echo V206_SERVICES_READY
    ;;
  *) echo "usage: $0 [start|service|bridge]" >&2; exit 2 ;;
esac
