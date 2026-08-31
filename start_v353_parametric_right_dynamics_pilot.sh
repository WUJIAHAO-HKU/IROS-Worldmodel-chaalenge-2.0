#!/usr/bin/env bash
set -euo pipefail
cd "/root/autodl-tmp/IROS_WAM_2.0 challenge"
run="artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822"
init="artifacts/strict_track2_joint_augmentation_20260810/v208_v205_mixed_right_gripper_contrast_long32_seed1407/selected_right_expert"
mkdir -p "$run"; export PYTHONPATH="pipeline:pipeline/scripts"; export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12; export MKL_NUM_THREADS=12
if [[ ! -f "$run/release_registration.json" ]]; then /root/miniconda3/envs/go1/bin/python pipeline/scripts/prepare_v353_parametric_right_dynamics_pilot.py; fi
exec taskset -c 0-11 /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_autoregressive_unet.py \
 --windows artifacts/adjust_bottle_windows_full --split-manifest "$run/public_right_only_split.json" --output "$run/model" \
 --init-autoregressive-checkpoint "$init" --normalization-checkpoint "$init" \
 --steps 100 --batch-size 1 --learning-rate 2e-7 --rollout-horizon 16 --train-rollout-steps 8 \
 --motion-weight 3.0 --motion-threshold 0.02 --horizon-loss-power 0.5 \
 --temporal-delta-weight 2.0 --temporal-delta-pool 4 --lowfreq-anchor-weight 1.0 --texture-laplacian-weight 0.1 \
 --high-motion-threshold 0.03 --high-motion-oversample-factor 2.0 --right-arm-oversample-factor 1.0 \
 --arm-sampling-source action --high-motion-selection-weight 0.5 \
 --validation-interval 20 --validation-batches 32 --checkpoint-interval 20 \
 --statistics-cache "$run/training_statistics.npz" --seed 1522 --device cuda --num-workers 2
