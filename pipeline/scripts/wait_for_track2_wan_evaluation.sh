#!/usr/bin/env bash
# Evaluate the immutable best Wan checkpoint only after the formal trainer exits.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

output_root="${WAM_WAN_OUTPUT:-artifacts/checkpoints/track2-wan-adjust-bottle-v1}"
base_model="${WAM_WAN_BASE_MODEL:-artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only}"
steps="${WAM_WAN_STEPS:-30000}"
windows="${WAM_WINDOWS:-artifacts/adjust_bottle_windows_full}"
split_manifest="${WAM_SPLIT_MANIFEST:-artifacts/splits/adjust_bottle_50episodes_full.json}"
best_evaluation="${WAM_WAN_EVALUATION:-artifacts/evaluations/track2_wan_adjust_bottle_best_validation_all.json}"
final_evaluation="${WAM_WAN_FINAL_EVALUATION:-artifacts/evaluations/track2_wan_adjust_bottle_final_validation_all.json}"
log_file="${WAM_WAN_POSTTRAIN_LOG:-artifacts/logs/track2_wan_posttrain.log}"
accept_mae="${WAM_WAN_ACCEPT_MAE:-1.0}"
inference_steps="${WAM_WAN_INFERENCE_STEPS:-30}"
evaluation_worker_log="${WAM_WAN_EVALUATION_WORKER_LOG:-artifacts/logs/track2_wan_evaluation_worker.log}"

[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "WAM_WAN_STEPS must be a positive integer" >&2; exit 2; }
[[ "$inference_steps" =~ ^[1-9][0-9]*$ ]] || { echo "WAM_WAN_INFERENCE_STEPS must be a positive integer" >&2; exit 2; }
python - "$accept_mae" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("WAM_WAN_ACCEPT_MAE must be a positive finite number") from exc
if not math.isfinite(value) or value <= 0:
    raise SystemExit("WAM_WAN_ACCEPT_MAE must be a positive finite number")
PY
mkdir -p "$(dirname "$log_file")"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$log_file"; }

completed_steps() {
  python - "$output_root/training_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    print(int(json.loads(path.read_text())["step"]))
except (OSError, ValueError, KeyError, json.JSONDecodeError):
    print(-1)
PY
}

while :; do
  current="$(completed_steps)"
  # The process check prevents evaluating while the final checkpoint is still
  # being written after its terminal validation pass.
  # Match this exact output path so an unrelated v1/v2 trainer cannot block or
  # prematurely release this post-training evaluator.
  if pgrep -f "[p]ython .*pipeline/scripts/train_track2_wan.py.*--output[[:space:]]+${output_root}" >/dev/null; then
    log "waiting for Wan trainer; checkpoint step=${current}/${steps}"
  elif [[ "$current" == "$steps" ]]; then
    break
  else
    # The low-priority worker can legitimately pause before Python has started
    # (or between resumed invocations) while another user owns the GPU.
    # Keep this post-training evaluator armed instead of misclassifying that
    # cooperative wait as a failed formal run.
    log "trainer is not active; awaiting formal resume at checkpoint step=${current}/${steps}"
  fi
  sleep 60
done

best="$output_root/best"
[[ -f "$best/track2_wan_lora.pt" && -f "$best/training_manifest.json" ]] || {
  log "final training completed but no validation-selected best checkpoint exists"
  exit 1
}
[[ -f "$output_root/track2_wan_lora.pt" && -f "$output_root/training_manifest.json" ]] || {
  log "final training completed but no final deployable checkpoint exists"
  exit 1
}

export PYTHONPATH="$root_dir/pipeline${PYTHONPATH:+:$PYTHONPATH}"

evaluate_candidate() {
  local checkpoint=$1
  local report=$2
  local label=$3
  log "evaluating ${label}: all 682 episode-held-out windows, RGB MAE threshold=${accept_mae}/255, flow steps=${inference_steps}"
  # Evaluation runs after training, but it still must yield GPU time to any
  # unrelated user workload that starts while the 682-window rollout runs.
  bash pipeline/scripts/opportunistic_gpu_worker.sh \
    --log "$evaluation_worker_log" --idle-seconds 30 --poll-seconds 5 --busy-sm 10 \
    --max-other-memory-mib 512 --min-free-mib 28000 --yield-on-other -- \
    conda run --no-capture-output -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
      --backend track2-wan \
      --windows "$windows" \
      --split-manifest "$split_manifest" \
      --checkpoint-dir "$checkpoint" \
      --wan-base-model "$base_model" \
      --wan-inference-steps "$inference_steps" \
      --split validation --samples 682 --accept-mae "$accept_mae" --require-pass \
      --output "$report"
}

write_acceptance() {
  local checkpoint=$1
  local report=$2
  conda run --no-capture-output -n go1 python pipeline/scripts/require_strict_world_model_acceptance.py \
    --evaluation "$report" \
    --windows "$windows" \
    --split-manifest "$split_manifest" \
    --checkpoint-dir "$checkpoint" \
    --backend track2-wan --accept-mae "$accept_mae" \
    --output "$checkpoint/strict_acceptance.json"
  log "strict Track 2 Wan acceptance completed for ${checkpoint}"
}

if evaluate_candidate "$best" "$best_evaluation" "validation-flow-best checkpoint ${best}"; then
  write_acceptance "$best" "$best_evaluation"
  exit 0
fi

log "validation-flow-best checkpoint missed the pixel gate; evaluating final step ${steps} checkpoint"
if evaluate_candidate "$output_root" "$final_evaluation" "final checkpoint ${output_root}"; then
  write_acceptance "$output_root" "$final_evaluation"
  exit 0
fi

log "both saved formal Wan candidates missed the all-window pixel gate"
exit 1
