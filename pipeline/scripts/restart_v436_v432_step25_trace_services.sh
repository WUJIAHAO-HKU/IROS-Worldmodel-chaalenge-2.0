#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"; GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
BASE="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v436_v432_step25_parent_diagnostic_release"
REG="${TRACK2_SERVICE_REG:?required}"; TRACE="${TRACK2_TRACE_DIR:?required}"; CPUSET="${TRACK2_CPUSET:-0-5}"; VERSION='track2-v436-v432-step25-parent-diagnostic'; TOKEN='local-dev-token'
case "${1:-start}" in
 service) cd "$ROOT"; exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$VERSION" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='v436-v432-step25-trace' WAM_CHECKPOINT_DIR="$BASE" WAM_V436_TRACE_DIR="$TRACE" WAM_DEVICE=cuda WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=1 "$GO1" pipeline/scripts/serve.py >>"$REG/v436_service.log" 2>&1;;
 bridge) cd "$ROOT"; exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$VERSION" --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$REG/v436_bridge.log" 2>&1;;
 stop) screen -S wm_v436_bridge -X quit >/dev/null 2>&1||true;screen -S wm_v436_gpu -X quit >/dev/null 2>&1||true;;
 start)
  mkdir -p "$REG" "$TRACE";test -z "$(find "$TRACE" -mindepth 1 -maxdepth 1 -print -quit)"
  for n in wm_v436_bridge wm_v436_gpu wm_v218_bridge wm_v218_gpu;do screen -S "$n" -X quit >/dev/null 2>&1||true;done
  for _ in $(seq 1 30);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
  screen -dmS wm_v436_gpu bash "$0" service
  for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&break;sleep 1;done
  curl -fsS http://127.0.0.1:8005/v1/health|grep -q "$VERSION";screen -dmS wm_v436_bridge bash "$0" bridge
  for _ in $(seq 1 60);do curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&break;sleep 1;done
  echo V436_TRACE_READY;;
 *)exit 2;;
esac
