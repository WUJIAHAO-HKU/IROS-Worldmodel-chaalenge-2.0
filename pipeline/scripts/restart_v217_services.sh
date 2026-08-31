#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
V209="$JOINT/v209_v202_v208_public_arm_routed_release"
V214="$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413"
V217="$JOINT/v217_v216_online_recursive_service_gate_seed1416"
MODEL_VERSION='track2-v217-public-knn-blend-alpha070-online-gated'
TOKEN='local-dev-token'
SERVICE_URL='http://127.0.0.1:8005'

run_service() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" \
    WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="$TOKEN" \
    WAM_BACKEND='v216-public-knn-blend' WAM_CHECKPOINT_DIR="$V209" \
    WAM_V216_LIBRARY_INDEX="$V214/library/public_right_knn.npz" \
    WAM_DEVICE='cuda' WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 \
    WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=0 \
    "$GO1" pipeline/scripts/serve.py >> "$V217/v217_service_gpu.log" 2>&1
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
  stop)
    screen -S wm_v217_gpu -X quit >/dev/null 2>&1 || true
    ;;
  start)
    screen -S wm_v217_gpu -X quit >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      if ! ss -ltn | grep -q ':8005 '; then break; fi
      sleep 1
    done
    if ss -ltn | grep -q ':8005 '; then
      echo 'v217 service port did not stop cleanly' >&2
      exit 3
    fi
    screen -dmS wm_v217_gpu bash "$0" service
    wait_for "$SERVICE_URL/v1/health" 120
    curl -fsS "$SERVICE_URL/v1/health" | grep -q "$MODEL_VERSION"
    curl -fsS -H "Authorization: Bearer $TOKEN" "$SERVICE_URL/v1/capabilities" | grep -q "$MODEL_VERSION"
    echo V217_SERVICE_READY
    ;;
  *) echo "usage: $0 [start|stop|service]" >&2; exit 2 ;;
esac
