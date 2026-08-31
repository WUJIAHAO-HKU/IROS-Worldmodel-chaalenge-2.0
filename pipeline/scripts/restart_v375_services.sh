#!/usr/bin/env bash
set -euo pipefail
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v375_bounded_cartesian_phase_pilot_seed1538_20260823"
RELEASE="$RUN/release"
LIBRARY="$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
VERSION='track2-v375-bounded-cartesian-phase-r16-p1'
TOKEN='local-dev-token'

service() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$VERSION" WAM_BEARER_TOKEN="$TOKEN" \
    WAM_BACKEND='v375-bounded-cartesian-phase' WAM_CHECKPOINT_DIR="$RELEASE" \
    WAM_V216_LIBRARY_INDEX="$LIBRARY" WAM_DEVICE=cuda WAM_PORT=8005 \
    WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=0 \
    "$GO1" pipeline/scripts/serve.py >>"$RUN/v375_service.log" 2>&1
}
bridge() {
  cd "$ROOT"
  exec env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server \
    --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$VERSION" \
    --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$RUN/v375_bridge.log" 2>&1
}
waitfor() { for _ in $(seq 1 "$2"); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
case "${1:-start}" in
  service) service ;;
  bridge) bridge ;;
  stop)
    screen -S wm_v375_bridge -X quit >/dev/null 2>&1 || true
    screen -S wm_v375_gpu -X quit >/dev/null 2>&1 || true
    ;;
  start)
    for name in wm_v375_bridge wm_v375_gpu wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
    for _ in $(seq 1 30); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
    ! ss -ltn | grep -qE ':(8005|18084) ' || exit 3
    screen -dmS wm_v375_gpu bash "$0" service
    waitfor http://127.0.0.1:8005/v1/health 120
    curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$VERSION"
    screen -dmS wm_v375_bridge bash "$0" bridge
    waitfor http://127.0.0.1:18084/health 60
    echo V375_SERVICES_READY
    ;;
  *) exit 2 ;;
esac
