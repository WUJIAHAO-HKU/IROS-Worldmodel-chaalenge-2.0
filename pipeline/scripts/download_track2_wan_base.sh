#!/usr/bin/env bash
# Download only the action-only Wan components needed by the Track 2 trainer.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [output-directory]" >&2
  exit 2
fi

output_dir="${1:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
revision="b8fff7315c768468a5333511427288870b2e9635"
endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"
workers="${WAM_WAN_DOWNLOAD_WORKERS:-12}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI 'hf' is required; install huggingface_hub first." >&2
  exit 1
fi

mkdir -p "$output_dir"
available_gb=$(df -Pk "$output_dir" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
if (( available_gb < 30 )); then
  echo "Need at least 30 GB free for the action-only Wan base; only ${available_gb} GB is available." >&2
  exit 1
fi

# Text encoder/tokenizer shards are intentionally omitted: Track 2 uses the
# learned abs14 conditioner as its only cross-attention input.
# Fetch small metadata through the Hub, then use bounded retryable ranges for
# the multi-GB model objects (see download_track2_wan_weights.py).
HF_ENDPOINT="$endpoint" HF_HUB_DISABLE_XET=1 hf download "$repo_id" \
  --revision "$revision" \
  --include 'transformer/config.json' \
  --include 'transformer/diffusion_pytorch_model.safetensors.index.json' \
  --include 'vae/config.json' \
  --local-dir "$output_dir" \
  --max-workers 2
python "$(dirname "$0")/download_track2_wan_weights.py" \
  --output "$output_dir" --endpoint "$endpoint" --workers "$workers"

for required in \
  transformer/config.json \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  vae/config.json \
  vae/diffusion_pytorch_model.safetensors; do
  if [[ ! -f "$output_dir/$required" ]]; then
    echo "Wan download is incomplete: missing $output_dir/$required" >&2
    exit 1
  fi
done

if [[ $(find "$output_dir/transformer" -maxdepth 1 -name 'diffusion_pytorch_model-*-of-*.safetensors' -type f | wc -l) -ne 5 ]]; then
  echo "Wan download is incomplete: expected five transformer shards." >&2
  exit 1
fi

echo "Track 2 Wan base ready: $output_dir"
