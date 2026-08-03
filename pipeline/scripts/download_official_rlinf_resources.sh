#!/usr/bin/env bash
# Download only the policy and reward paths published for Track 2.
set -euo pipefail

resources=${1:-artifacts/official_resources}
root_dir=$(cd "$(dirname "$0")/../.." && pwd)
resources=$(cd "$root_dir" && mkdir -p "$resources" && cd "$resources" && pwd)
mkdir -p "$resources"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

hf download WorldArena/WorldArena2.0 --include 'pi05_adjust_bottle/**' --local-dir "$resources" --max-workers 2
hf download WorldArena/WorldArena2.0 --include 'reward_model/adjust_bottle/full_weights.pt' --local-dir "$resources" --max-workers 2
hf download google-t5/t5-base config.json spiece.model model.safetensors \
  --local-dir "$resources/reward_model/t5-base" --max-workers 2
