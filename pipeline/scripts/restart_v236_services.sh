#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
V209="$JOINT/v209_v202_v208_public_arm_routed_release"
V214="$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413"
V236="$JOINT/v236_v235c_causal_right_close_service_seed1435_20260818"
MODEL_VERSION='track2-v236-public-knn-alpha070-causal-right-close'
TOKEN='local-dev-token'
SERVICE_URL='http://127.0.0.1:8005'
BRIDGE_URL='http://127.0.0.1:18084'

run_service() {
  local service_device="${TRACK2_WAM_DEVICE:-cuda}"
  local release_cuda_cache="${TRACK2_WAM_RELEASE_CUDA_CACHE:-0}"
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" \
    WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="$TOKEN" \
    WAM_BACKEND='v236-causal-public-knn-blend' WAM_CHECKPOINT_DIR="$V209" \
    WAM_V216_LIBRARY_INDEX="$V214/library/public_right_knn.npz" \
    WAM_DEVICE="$service_device" WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 \
    WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE="$release_cuda_cache" \
    "$GO1" pipeline/scripts/serve.py >>"$V236/v236_service.log" 2>&1
}

run_bridge() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server \
    --world-model-url "$SERVICE_URL" --token "$TOKEN" \
    --model-version "$MODEL_VERSION" --host 127.0.0.1 --port 18084 \
    --audit-max-items "${TRACK2_BRIDGE_AUDIT_MAX_ITEMS:-1}" \
    >>"$V236/v236_bridge.log" 2>&1
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
  stop)
    screen -S wm_v236_bridge -X quit >/dev/null 2>&1 || true
    screen -S wm_v236_gpu -X quit >/dev/null 2>&1 || true
    ;;
  start)
    for name in wm_v236_bridge wm_v236_gpu wm_v218_bridge wm_v218_gpu; do
      screen -S "$name" -X quit >/dev/null 2>&1 || true
    done
    for _ in $(seq 1 30); do
      if ! ss -ltn | grep -qE ':(8005|18084) '; then break; fi
      sleep 1
    done
    if ss -ltn | grep -qE ':(8005|18084) '; then
      echo 'v236 ports did not stop cleanly' >&2
      exit 3
    fi
    screen -dmS wm_v236_gpu bash "$0" service
    wait_for "$SERVICE_URL/v1/health" 120
    curl -fsS "$SERVICE_URL/v1/health" | grep -q "$MODEL_VERSION"
    screen -dmS wm_v236_bridge bash "$0" bridge
    wait_for "$BRIDGE_URL/health" 60
    curl -fsS "$BRIDGE_URL/health" | grep -q '"status":"ready"'
    echo V236_SERVICES_READY
    ;;
  *) echo "usage: $0 [start|stop|service|bridge]" >&2; exit 2 ;;
esac
