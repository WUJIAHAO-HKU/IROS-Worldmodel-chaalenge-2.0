#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v360_v355_service_acceptance_seed1527_20260822"
candidate="artifacts/strict_track2_joint_augmentation_20260810/v355_v202_v354_parametric_arm_routed_release"
mkdir -p "$run"; export PYTHONPATH="pipeline:pipeline/scripts"
if [[ ! -f "$run/release_registration.json" ]]; then /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v360_v355_service_acceptance.py; fi
token="$(<"$run/bearer_token.txt")"
screen -S wm_v355_service -X quit >/dev/null 2>&1 || true
screen -dmS wm_v355_service bash -lc "cd '/root/autodl-tmp/IROS_WAM_2.0 challenge' && export PYTHONPATH='pipeline:pipeline/scripts' CUDA_VISIBLE_DEVICES=0 WAM_PORT=18055 WAM_MODEL_VERSION='track2-v355-v202-left-v354-parametric-right' WAM_BEARER_TOKEN='$token' WAM_BACKEND='arm-routed-autoregressive-unet' WAM_CHECKPOINT_DIR='$candidate' WAM_DEVICE=cuda WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 WAM_BATCH_WORKERS=1 WAM_IMAGE_CODEC_WORKERS=8 && exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python pipeline/scripts/serve.py > '$run/service.log' 2>&1"
ready=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:18055/v1/health >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then tail -n 100 "$run/service.log"; exit 1; fi
exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python pipeline/scripts/strict_service_acceptance.py \
 --base-url http://127.0.0.1:18055 --token-file "$run/bearer_token.txt" \
 --model-version track2-v355-v202-left-v354-parametric-right --output "$run/acceptance_report.json"
