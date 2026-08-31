#!/usr/bin/env bash
# Second motion/texture pilot: suppress noisy pixel deltas with 4x4 temporal pooling.
set -euo pipefail

export WAM_MOTION_TEXTURE_OUTPUT="${WAM_MOTION_TEXTURE_OUTPUT:-artifacts/checkpoints/autoregressive-unet-track2-rollout8-motion-texture-coarse-refine-v2}"
export WAM_MOTION_TEXTURE_CANDIDATE="${WAM_MOTION_TEXTURE_CANDIDATE:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_coarse_refine_preview64.json}"
export WAM_MOTION_TEXTURE_PREDICTIONS="${WAM_MOTION_TEXTURE_PREDICTIONS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_coarse_refine_preview64_predictions.npz}"
export WAM_MOTION_TEXTURE_DYNAMICS="${WAM_MOTION_TEXTURE_DYNAMICS:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_coarse_refine_preview64_dynamics.json}"
export WAM_MOTION_TEXTURE_COMPARISON="${WAM_MOTION_TEXTURE_COMPARISON:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_coarse_refine_preview64_comparison.json}"
export WAM_MOTION_TEXTURE_PROMOTION="${WAM_MOTION_TEXTURE_PROMOTION:-artifacts/evaluations/autoregressive_unet_rollout8_motion_texture_coarse_refine_preview64_promotion.json}"
export WAM_MOTION_TEXTURE_STEPS="${WAM_MOTION_TEXTURE_STEPS:-500}"
export WAM_MOTION_TEXTURE_BATCH_SIZE="${WAM_MOTION_TEXTURE_BATCH_SIZE:-10}"
export WAM_MOTION_TEXTURE_LEARNING_RATE="${WAM_MOTION_TEXTURE_LEARNING_RATE:-3e-7}"
export WAM_TEMPORAL_DELTA_WEIGHT="${WAM_TEMPORAL_DELTA_WEIGHT:-0.25}"
export WAM_TEXTURE_LAPLACIAN_WEIGHT="${WAM_TEXTURE_LAPLACIAN_WEIGHT:-0.06}"
export WAM_TEMPORAL_DELTA_POOL="${WAM_TEMPORAL_DELTA_POOL:-4}"
export WAM_MOTION_TEXTURE_SEED="${WAM_MOTION_TEXTURE_SEED:-20260812}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_autoregressive_motion_texture_refine.sh"
