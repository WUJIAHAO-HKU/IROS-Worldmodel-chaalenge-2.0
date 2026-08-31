#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v357_v354s150_temporal_highmotion_refine_seed1525_20260822"
parent="artifacts/strict_track2_joint_augmentation_20260810/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
split="artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822/public_right_only_split.json"
stats="artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822/training_statistics.npz"
mkdir -p "$run"; export PYTHONPATH="pipeline:pipeline/scripts"; export CUDA_VISIBLE_DEVICES=0; export OMP_NUM_THREADS=12; export MKL_NUM_THREADS=12
if [[ ! -f "$run/release_registration.json" ]]; then /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v357_temporal_highmotion_refine.py; fi
exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
 --windows artifacts/adjust_bottle_windows_full --split-manifest "$split" --output "$run/model" \
 --init-autoregressive-checkpoint "$parent" --normalization-checkpoint "$parent" \
 --steps 200 --batch-size 1 --learning-rate 2e-7 --rollout-horizon 16 --train-rollout-steps 8 \
 --motion-weight 4.0 --motion-threshold 0.02 --horizon-loss-power 0.75 --temporal-delta-weight 4.0 --temporal-delta-pool 4 \
 --lowfreq-anchor-weight 2.0 --texture-laplacian-weight 0.05 --high-motion-threshold 0.03 --high-motion-oversample-factor 4.0 \
 --right-arm-oversample-factor 1.0 --arm-sampling-source action --high-motion-selection-weight 1.0 \
 --validation-interval 40 --validation-batches 32 --checkpoint-interval 40 \
 --statistics-cache "$stats" --seed 1525 --device cuda --num-workers 2
