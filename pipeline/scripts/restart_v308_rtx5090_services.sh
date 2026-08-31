#!/usr/bin/env bash
set -euo pipefail

# Blackwell changes batched convolution numerics when TF32 is enabled.  The
# frozen v303 numeric gate is recovered by disabling TF32 without changing
# model weights, inputs, output ordering, or API behavior.
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec bash '/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/restart_v301_services.sh' "${1:-start}"
