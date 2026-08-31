#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}" WAM_IMAGE_CODEC_WORKERS="${WAM_IMAGE_CODEC_WORKERS:-6}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
R="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
P="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
O="$R/artifacts/strict_track2_official_20260810"
REG="${TRACK2_SERVICE_REG:-$O/run_registry/v361_v169_v355_rolloutonly32_seed1528_20260822}"
V='track2-v355-v202-left-v354-parametric-right'
TOKEN='local-dev-token'
CPUSET="${TRACK2_CPUSET:-0-5}"
CANDIDATE="$J/v355_v202_v354_parametric_arm_routed_release"

svc() {
  cd "$R"
  exec taskset -c "$CPUSET" env PYTHONPATH="$R/pipeline" WAM_MODEL_VERSION="$V" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='arm-routed-autoregressive-unet' WAM_CHECKPOINT_DIR="$CANDIDATE" WAM_DEVICE="${TRACK2_WAM_DEVICE:-cuda}" WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE="${TRACK2_WAM_RELEASE_CUDA_CACHE:-1}" "$P" pipeline/scripts/serve.py >>"$REG/v355_service.log" 2>&1
}

bridge() {
  cd "$R"
  exec taskset -c "$CPUSET" env PYTHONPATH="$R/pipeline" "$P" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$V" --host 127.0.0.1 --port 18085 --audit-max-items 1 >>"$REG/v355_bridge.log" 2>&1
}

waitf() {
  for _ in $(seq 1 "$2"); do
    curl -fsS "$1" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

case "${1:-start}" in
  svc) svc ;;
  bridge) bridge ;;
  stop)
    screen -S wm_v355_bridge -X quit >/dev/null 2>&1 || true
    screen -S wm_v355_gpu -X quit >/dev/null 2>&1 || true
    screen -S wm_v355_service -X quit >/dev/null 2>&1 || true
    ;;
  start)
    mkdir -p "$REG"
    for n in wm_v355_bridge wm_v355_gpu wm_v355_service wm_v326_bridge wm_v326_gpu wm_v317_bridge wm_v317_gpu; do
      screen -S "$n" -X quit >/dev/null 2>&1 || true
    done
    for _ in $(seq 1 30); do
      ! ss -ltn | grep -qE ':(8005|18085) ' && break
      sleep 1
    done
    screen -dmS wm_v355_gpu bash "$0" svc
    waitf http://127.0.0.1:8005/v1/health 120
    curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$V"
    screen -dmS wm_v355_bridge bash "$0" bridge
    waitf http://127.0.0.1:18085/health 60
    echo V355_SERVICES_READY
    ;;
  *) exit 2 ;;
esac
