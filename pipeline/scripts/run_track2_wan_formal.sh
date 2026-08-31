#!/usr/bin/env bash
# Validate, smoke-test, cache, and then start/resume formal Track 2 Wan LoRA training.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

dataset_root="${WAM_WAN_DATASET_ROOT:-artifacts/wan/adjust_bottle_rlinf_npy_v1}"
base_model="${WAM_WAN_BASE_MODEL:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
cache_root="${WAM_WAN_LATENT_CACHE:-artifacts/wan/adjust_bottle_wan22_latents_v1}"
output_root="${WAM_WAN_OUTPUT:-artifacts/checkpoints/track2-wan-adjust-bottle-v1}"
steps="${WAM_WAN_STEPS:-30000}"
resume="${WAM_WAN_RESUME:-}"
init_checkpoint="${WAM_WAN_INIT_CHECKPOINT:-}"
trajectory_conditioned="${WAM_WAN_TRAJECTORY_CONDITIONED:-0}"
latent_action_conditioned="${WAM_WAN_LATENT_ACTION_CONDITIONED:-0}"
high_motion_oversample="${WAM_WAN_HIGH_MOTION_OVERSAMPLE:-0}"
trajectory_warmup_steps="${WAM_WAN_TRAJECTORY_WARMUP_STEPS:-0}"
freeze_trajectory_during_warmup="${WAM_WAN_FREEZE_TRAJECTORY_DURING_WARMUP:-0}"
trajectory_hidden_dim="${WAM_WAN_TRAJECTORY_HIDDEN_DIM:-}"
latent_action_hidden_dim="${WAM_WAN_LATENT_ACTION_HIDDEN_DIM:-}"
learning_rate="${WAM_WAN_LEARNING_RATE:-1e-4}"
inherited_learning_rate="${WAM_WAN_INHERITED_LEARNING_RATE:-}"
validation_interval="${WAM_WAN_VALIDATION_INTERVAL:-500}"
save_interval="${WAM_WAN_SAVE_INTERVAL:-500}"
num_workers="${WAM_WAN_NUM_WORKERS:-2}"
seed="${WAM_WAN_SEED:-0}"
smoke_output="${WAM_WAN_SMOKE_OUTPUT:-artifacts/smoke/track2-wan-real}"

# A checkpoint directory has one writer. Holding this descriptor through the
# final exec prevents duplicate launchers from racing on best/ and root files.
mkdir -p "$output_root"
exec 9>"$output_root/.trainer.lock"
if ! flock -n 9; then
  echo "another Track 2 Wan trainer already owns: $output_root" >&2
  exit 1
fi
if [[ "$trajectory_conditioned" == "1" && "$latent_action_conditioned" == "1" && -z "${WAM_WAN_SMOKE_OUTPUT:-}" ]]; then
  smoke_output="artifacts/smoke/track2-wan-trajectory-latent-v4-real"
elif [[ "$latent_action_conditioned" == "1" && -z "${WAM_WAN_SMOKE_OUTPUT:-}" ]]; then
  smoke_output="artifacts/smoke/track2-wan-latent-action-v3-real"
elif [[ "$trajectory_conditioned" == "1" && -z "${WAM_WAN_SMOKE_OUTPUT:-}" ]]; then
  smoke_output="artifacts/smoke/track2-wan-trajectory-v2-real"
fi

for required in \
  "$base_model/transformer/config.json" \
  "$base_model/transformer/diffusion_pytorch_model.safetensors.index.json" \
  "$base_model/vae/config.json" \
  "$base_model/vae/diffusion_pytorch_model.safetensors"; do
  [[ -f "$required" ]] || { echo "required Wan file is missing: $required" >&2; exit 1; }
done
for shard in 1 2 3 4 5; do
  [[ -f "$base_model/transformer/diffusion_pytorch_model-0000${shard}-of-00005.safetensors" ]] || {
    echo "required Wan transformer shard is missing: $shard" >&2
    exit 1
  }
done

# File existence alone cannot prove a resumed multi-GB download is valid.
conda run --no-capture-output -n go1 python pipeline/scripts/download_track2_wan_weights.py \
  --output "$base_model" --verify-only

if [[ -n "$resume" && -n "$init_checkpoint" ]]; then
  echo "WAM_WAN_RESUME and WAM_WAN_INIT_CHECKPOINT cannot both be set" >&2
  exit 2
fi
# A cooperative GPU worker can terminate this process after an unrelated job
# arrives. Always resume an existing output checkpoint instead of replaying
# --init-checkpoint and overwriting completed progress.
if [[ -z "$resume" && -f "$output_root/track2_wan_lora.pt" ]]; then
  resume="$output_root/track2_wan_lora.pt"
  init_checkpoint=""
fi

export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"
conda run --no-capture-output -n go1 python pipeline/scripts/validate_wan_track2_windows.py \
  --dataset-root "$dataset_root" \
  --report artifacts/wan/track2_wan_window_validation_latest.json
smoke_args=(
  pipeline/scripts/smoke_track2_wan_real.py
  --dataset-root "$dataset_root" \
  --base-model "$base_model" \
  --output "$smoke_output"
)
if [[ "$trajectory_conditioned" == "1" ]]; then
  smoke_args+=(--trajectory-conditioned)
fi
if [[ "$latent_action_conditioned" == "1" ]]; then
  smoke_args+=(--latent-action-conditioned)
fi
conda run --no-capture-output -n go1 python "${smoke_args[@]}"
conda run --no-capture-output -n go1 python pipeline/scripts/cache_track2_wan_latents.py \
  --dataset-root "$dataset_root" \
  --base-model "$base_model" \
  --output "$cache_root" --batch-size 4

train_args=(
  pipeline/scripts/train_track2_wan.py
  --dataset-root "$dataset_root"
  --base-model "$base_model"
  --output "$output_root"
  --latent-cache "$cache_root"
  --steps "$steps" --batch-size 1 --gradient-accumulation 4 --lora-rank 32
  --learning-rate "$learning_rate" --validation-interval "$validation_interval" --save-interval "$save_interval"
  --num-workers "$num_workers" --seed "$seed"
)
if [[ -n "$resume" ]]; then
  train_args+=(--resume "$resume")
fi
if [[ -n "$init_checkpoint" ]]; then
  train_args+=(--init-checkpoint "$init_checkpoint")
fi
if [[ "$trajectory_conditioned" == "1" ]]; then
  train_args+=(--trajectory-conditioned)
fi
if [[ "$latent_action_conditioned" == "1" ]]; then
  train_args+=(--latent-action-conditioned)
fi
if [[ -n "$trajectory_hidden_dim" ]]; then
  train_args+=(--trajectory-hidden-dim "$trajectory_hidden_dim")
fi
if [[ -n "$latent_action_hidden_dim" ]]; then
  train_args+=(--latent-action-hidden-dim "$latent_action_hidden_dim")
fi
if [[ -n "$inherited_learning_rate" ]]; then
  train_args+=(--inherited-learning-rate "$inherited_learning_rate")
fi
if [[ "$high_motion_oversample" != "0" ]]; then
  train_args+=(--high-motion-oversample "$high_motion_oversample")
fi
if [[ "$freeze_trajectory_during_warmup" == "1" ]]; then
  train_args+=(--freeze-trajectory-during-warmup)
fi
if [[ -z "$resume" && "$trajectory_warmup_steps" != "0" ]]; then
  # A resumed v2 checkpoint owns its global warmup boundary.  Passing this
  # flag again makes the trainer reject an otherwise compatible resume.
  train_args+=(--trajectory-warmup-steps "$trajectory_warmup_steps")
fi
exec conda run --no-capture-output -n go1 python "${train_args[@]}"
