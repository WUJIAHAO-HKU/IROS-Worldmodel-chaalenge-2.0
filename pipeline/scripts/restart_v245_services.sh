#!/usr/bin/env bash
set -euo pipefail
ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}";GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}";JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";V209="$JOINT/v209_v202_v208_public_arm_routed_release";V214="$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413";V245="$JOINT/v245b_v244_clean_progressive_service_seed1448_20260819";MODEL_VERSION='track2-v245b-public-clean-progressive-offset24';TOKEN='local-dev-token';SERVICE_URL='http://127.0.0.1:8005';BRIDGE_URL='http://127.0.0.1:18084'
run_service(){ cd "$ROOT";exec env PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="$TOKEN" WAM_BACKEND='v245-clean-progressive-successor' WAM_CHECKPOINT_DIR="$V209" WAM_V216_LIBRARY_INDEX="$V214/library/public_right_knn.npz" WAM_DEVICE="${TRACK2_WAM_DEVICE:-cuda}" WAM_PORT=8005 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_RELEASE_CUDA_CACHE=0 "$GO1" pipeline/scripts/serve.py >>"$V245/v245_service.log" 2>&1;}
run_bridge(){ cd "$ROOT";exec env PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server --world-model-url "$SERVICE_URL" --token "$TOKEN" --model-version "$MODEL_VERSION" --host 127.0.0.1 --port 18084 --audit-max-items 1 >>"$V245/v245_bridge.log" 2>&1;}
wait_for(){ for _ in $(seq 1 "$2");do curl -fsS "$1" >/dev/null 2>&1&&return 0;sleep 1;done;return 1;}
case "${1:-start}" in
 service)run_service;;bridge)run_bridge;;
 stop)screen -S wm_v245_bridge -X quit >/dev/null 2>&1||true;screen -S wm_v245_gpu -X quit >/dev/null 2>&1||true;;
 start)
  for n in wm_v245_bridge wm_v245_gpu wm_v241_bridge wm_v241_gpu wm_v236_bridge wm_v236_gpu wm_v218_bridge wm_v218_gpu;do screen -S "$n" -X quit >/dev/null 2>&1||true;done
  for _ in $(seq 1 30);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done
  ! ss -ltn|grep -qE ':(8005|18084) '||{ echo 'v245 ports did not stop cleanly' >&2;exit 3;}
  screen -dmS wm_v245_gpu bash "$0" service;wait_for "$SERVICE_URL/v1/health" 120;curl -fsS "$SERVICE_URL/v1/health"|grep -q "$MODEL_VERSION"
  screen -dmS wm_v245_bridge bash "$0" bridge;wait_for "$BRIDGE_URL/health" 60;echo V245_SERVICES_READY;;
 *)echo 'usage: restart_v245_services.sh [start|stop|service|bridge]' >&2;exit 2;;esac
