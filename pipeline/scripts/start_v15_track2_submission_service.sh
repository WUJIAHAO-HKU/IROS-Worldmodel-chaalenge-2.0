#!/usr/bin/env bash
set -euo pipefail

# AutoDL terminates public TLS and forwards the official HTTPS endpoint to the
# container's HTTP port 6006. The application must still enforce Bearer auth.
RUNTIME_ROOT="${WAM_RUNTIME_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
TOKEN_FILE="${WAM_TOKEN_FILE:-/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/service_acceptance/secrets/bearer_token}"
CHECKPOINT_DIR="${WAM_CHECKPOINT_DIR:-/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v157_hybrid_gate_blend12_formal_release}"
V15_LIBRARY_DIR="${WAM_V15_LIBRARY_DIR:-/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts}"

if [[ ! -r "$TOKEN_FILE" ]]; then
  printf 'Bearer token file is missing or unreadable: %s\n' "$TOKEN_FILE" >&2
  exit 2
fi

token_mode="$(stat -c '%a' "$TOKEN_FILE")"
if [[ "$token_mode" != "600" ]]; then
  printf 'Bearer token file must have mode 600, got %s\n' "$token_mode" >&2
  exit 2
fi

export WAM_BACKEND="${WAM_BACKEND:-v15-gated-composite}"
export WAM_BEARER_TOKEN
WAM_BEARER_TOKEN="$(<"$TOKEN_FILE")"
export WAM_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export WAM_DEVICE="${WAM_DEVICE:-cuda}"
export WAM_MODEL_VERSION="${WAM_MODEL_VERSION:-track2-v15.7-hybrid-action-gated-reward-safe-blend12}"
export WAM_PORT="${WAM_PORT:-6006}"
export WAM_V15_LIBRARY_DIR="$V15_LIBRARY_DIR"
export WAM_NATIVE_BATCH_ENABLED="${WAM_NATIVE_BATCH_ENABLED:-1}"
export WAM_NATIVE_BATCH_MICRO_SIZE="${WAM_NATIVE_BATCH_MICRO_SIZE:-8}"
export WAM_RELEASE_CUDA_CACHE="${WAM_RELEASE_CUDA_CACHE:-1}"
export PYTHONPATH="$RUNTIME_ROOT/pipeline"

cd "$RUNTIME_ROOT"
exec /root/miniconda3/envs/go1/bin/python pipeline/scripts/serve.py
