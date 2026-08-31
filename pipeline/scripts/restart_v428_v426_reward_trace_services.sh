#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 RAYON_NUM_THREADS=2 WAM_IMAGE_CODEC_WORKERS=6
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
BASE="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v426_v423_motion_guarded_right_release"
REG="${TRACK2_SERVICE_REG:?TRACK2_SERVICE_REG required}"
TRACE_DIR="${TRACK2_TRACE_DIR:?TRACK2_TRACE_DIR required}"
CPUSET="${TRACK2_CPUSET:-0-5}"
VERSION='track2-v428-v426-output-equivalent-reward-trace'
TOKEN='local-dev-token'
service() {
  cd "$ROOT"
  exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$VERSION" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='v428-v426-reward-trace' WAM_CHECKPOINT_DIR="$BASE" WAM_V428_TRACE_DIR="$TRACE_DIR" WAM_DEVICE=cuda WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=1 "$GO1" pipeline/scripts/serve.py >>"$REG/v428_service.log" 2>&1
}
bridge() {
  cd "$ROOT"
  exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$VERSION" --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$REG/v428_bridge.log" 2>&1
}
waitfor() { for _ in $(seq 1 "$2"); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
case "${1:-start}" in
  service) service ;;
  bridge) bridge ;;
  stop) screen -S wm_v428_bridge -X quit >/dev/null 2>&1 || true; screen -S wm_v428_gpu -X quit >/dev/null 2>&1 || true ;;
  start)
    mkdir -p "$REG" "$TRACE_DIR"
    test -z "$(find "$TRACE_DIR" -mindepth 1 -maxdepth 1 -print -quit)"
    for name in wm_v428_bridge wm_v428_gpu wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
    for _ in $(seq 1 30); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
    screen -dmS wm_v428_gpu bash "$0" service
    waitfor http://127.0.0.1:8005/v1/health 120
    curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$VERSION"
    screen -dmS wm_v428_bridge bash "$0" bridge
    waitfor http://127.0.0.1:18084/health 60
    echo V428_V426_REWARD_TRACE_SERVICES_READY ;;
  *) exit 2 ;;
esac
