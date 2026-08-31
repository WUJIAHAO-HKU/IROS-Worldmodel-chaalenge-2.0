#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}" WAM_IMAGE_CODEC_WORKERS="${WAM_IMAGE_CODEC_WORKERS:-6}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"; GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"; J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$J/v385_native_batch_clean_reanchor_seed1548_20260823"; REG="${TRACK2_SERVICE_REG:-$RUN}"; CPUSET="${TRACK2_CPUSET:-0-5}"; RELEASE="$RUN/release"
LIBRARY="$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"; VERSION='track2-v385-v326-nativebatch-actionphase-clean-reanchor'; TOKEN='local-dev-token'
ACTION_GATE="$J/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz"; PHASE_GATE="$J/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz"
service(){ cd "$ROOT"; exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$VERSION" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='v385-native-batch-clean-reanchor' WAM_CHECKPOINT_DIR="$RELEASE" WAM_V216_LIBRARY_INDEX="$LIBRARY" WAM_V312_ACTION_GATE="$ACTION_GATE" WAM_V324_PHASE_GATE="$PHASE_GATE" WAM_DEVICE="${TRACK2_WAM_DEVICE:-cuda}" WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE="${TRACK2_WAM_RELEASE_CUDA_CACHE:-1}" "$GO1" pipeline/scripts/serve.py >>"$REG/v385_service.log" 2>&1; }
bridge(){ cd "$ROOT"; exec taskset -c "$CPUSET" env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$VERSION" --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$REG/v385_bridge.log" 2>&1; }
waitfor(){ for _ in $(seq 1 "$2"); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
case "${1:-start}" in
  service) service ;;
  bridge) bridge ;;
  stop) screen -S wm_v385_bridge -X quit >/dev/null 2>&1 || true; screen -S wm_v385_gpu -X quit >/dev/null 2>&1 || true ;;
  start)
    mkdir -p "$REG"
    for name in wm_v385_bridge wm_v385_gpu wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
    for _ in $(seq 1 30); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
    ! ss -ltn | grep -qE ':(8005|18084) ' || exit 3
    screen -dmS wm_v385_gpu bash "$0" service; waitfor http://127.0.0.1:8005/v1/health 120
    curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$VERSION"
    screen -dmS wm_v385_bridge bash "$0" bridge; waitfor http://127.0.0.1:18084/health 60
    echo V385_SERVICES_READY ;;
  *) exit 2 ;;
esac
