#!/usr/bin/env bash
set -euo pipefail
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}" TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export WAM_IMAGE_CODEC_WORKERS="${WAM_IMAGE_CODEC_WORKERS:-8}"
R="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
P="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
J="$R/artifacts/strict_track2_joint_augmentation_20260810"
O="$R/artifacts/strict_track2_official_20260810"
REG="${TRACK2_SERVICE_REG:-$O/run_registry/v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819}"
V='track2-v271-endpoint-calibrated-terminal'
TOKEN='local-dev-token'
CPUSET="${TRACK2_CPUSET:-0-5}"
svc(){ cd "$R"; exec taskset -c "$CPUSET" env PYTHONPATH="$R/pipeline" WAM_MODEL_VERSION="$V" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='v271-endpoint-calibrated-terminal' WAM_CHECKPOINT_DIR="$J/v209_v202_v208_public_arm_routed_release" WAM_V216_LIBRARY_INDEX="$J/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" WAM_DEVICE="${TRACK2_WAM_DEVICE:-cuda}" WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE="${TRACK2_WAM_RELEASE_CUDA_CACHE:-1}" "$P" pipeline/scripts/serve.py >>"$REG/v271_service.log" 2>&1; }
bridge(){ cd "$R"; exec taskset -c "$CPUSET" env PYTHONPATH="$R/pipeline" "$P" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8005 --token "$TOKEN" --model-version "$V" --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$REG/v271_bridge.log" 2>&1; }
waitf(){ for _ in $(seq 1 "$2"); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
case "${1:-start}" in
  svc) svc ;;
  bridge) bridge ;;
  stop) screen -S wm_v271_v274_bridge -X quit >/dev/null 2>&1 || true; screen -S wm_v271_v274_gpu -X quit >/dev/null 2>&1 || true ;;
  start)
    for n in wm_v271_v274_bridge wm_v271_v274_gpu wm_v218_bridge wm_v218_gpu; do screen -S "$n" -X quit >/dev/null 2>&1 || true; done
    for _ in $(seq 1 30); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
    screen -dmS wm_v271_v274_gpu bash "$0" svc
    waitf http://127.0.0.1:8005/v1/health 120
    curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$V"
    screen -dmS wm_v271_v274_bridge bash "$0" bridge
    waitf http://127.0.0.1:18084/health 60
    echo V271_V274_SERVICES_READY ;;
  *) exit 2 ;;
esac
