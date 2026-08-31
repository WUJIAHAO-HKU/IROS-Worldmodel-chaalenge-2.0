#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"

registry="artifacts/strict_track2_official_20260810/run_registry/v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823"
work="/dev/shm/v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823"
parent="artifacts/strict_track2_joint_augmentation_20260810/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
split="artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"

mkdir -p "$registry" "$work"
export PYTHONPATH="pipeline:pipeline/scripts"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

if [[ ! -f "$registry/preregistration.json" ]]; then
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v423_mirror_augmented_right_dynamics.py
fi

exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/train_v423_mirror_augmented_autoregressive_unet.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest "$split" \
  --output "$work/model" \
  --init-autoregressive-checkpoint "$parent" \
  --normalization-checkpoint "$parent" \
  --steps 100 --batch-size 1 --learning-rate 2e-7 \
  --rollout-horizon 32 --train-rollout-steps 16 \
  --motion-weight 3.0 --motion-threshold 0.02 --horizon-loss-power 0.75 \
  --temporal-delta-weight 2.0 --temporal-delta-pool 4 \
  --lowfreq-anchor-weight 1.0 --texture-laplacian-weight 0.1 \
  --high-motion-threshold 0.03 --high-motion-oversample-factor 3.0 \
  --right-arm-oversample-factor 1.0 --arm-sampling-source action \
  --high-motion-selection-weight 0.75 \
  --validation-interval 25 --validation-batches 32 --checkpoint-interval 25 \
  --early-stop-validations 2 --minimum-relative-improvement 0.002 \
  --statistics-cache "$work/training_statistics.npz" \
  --seed 1575 --device cuda --num-workers 2
