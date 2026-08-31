#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
ROBOTWIN_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
OPENPI_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/openpi-rlinf-full"
AUDIT_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810"
EVAL_ROOT="$AUDIT_ROOT/real_robotwin_eval"
INSTRUMENTED_ROOT="$EVAL_ROOT/instrumented_runtime"
DEV_ROOT="${TRACK2_DEV_SEED_ROOT:-$EVAL_ROOT/development_seeds22}"
OUTPUT_ROOT="${TRACK2_DEV_OUTPUT_ROOT:-$EVAL_ROOT/development_metrics}"
MODEL_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/pi05_adjust_bottle"
PYTHON_BIN="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
CANDIDATE_CHECKPOINT="${TRACK2_CANDIDATE_CHECKPOINT:-}"
CANDIDATE_VARIANT="${TRACK2_CANDIDATE_VARIANT:-}"
BATCH_FILTER="${TRACK2_DEV_BATCH_FILTER:-}"
SKIP_BASELINE="${TRACK2_SKIP_BASELINE:-false}"
BASELINE_ONLY="${TRACK2_EVAL_BASELINE_ONLY:-false}"
if [[ "$BASELINE_ONLY" != "true" && "$BASELINE_ONLY" != "false" ]]; then
  printf 'TRACK2_EVAL_BASELINE_ONLY must be true or false\n' >&2
  exit 2
fi
if [[ "$BASELINE_ONLY" != "true" && ( -z "$CANDIDATE_CHECKPOINT" || -z "$CANDIDATE_VARIANT" ) ]]; then
  printf 'set TRACK2_CANDIDATE_CHECKPOINT and TRACK2_CANDIDATE_VARIANT unless baseline-only\n' >&2
  exit 2
fi
if [[ -n "$BATCH_FILTER" && ! "$BATCH_FILTER" =~ ^[0-9][0-9]$ ]]; then
  printf 'TRACK2_DEV_BATCH_FILTER must be a two-digit batch ID\n' >&2
  exit 2
fi

run_eval() {
  local variant="$1"
  local batch="$2"
  local total="$3"
  local checkpoint="$4"
  local run_dir="$OUTPUT_ROOT/$variant/batch_$batch"
  local log_path="$run_dir/launcher.log"
  local seed_path="$DEV_ROOT/batch_$batch.json"

  if grep -aq "'eval/num_trajectories': $total" "$log_path" 2>/dev/null \
    && grep -aq "'eval/grasp_once'" "$log_path" 2>/dev/null \
    && grep -aq "'eval/success_bitmask'" "$log_path" 2>/dev/null; then
    printf 'SKIP variant=%s batch=%s reason=complete\n' "$variant" "$batch"
    return
  fi

  "$PYTHON_BIN" -m ray.scripts.scripts stop --force \
    >"/tmp/ray_stop_track2_dev_${variant}_${batch}.log" 2>&1 || true
  mkdir -p "$run_dir"
  cd /tmp

  REPO_PATH="$RLINF_ROOT" \
  PYTHONPATH="$INSTRUMENTED_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$ROBOTWIN_ROOT:$RLINF_ROOT" \
  PYTHONHASHSEED=0 \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$PYTHON_BIN" -c \
    'import rlinf.envs.robotwin.robotwin_env as r; import rlinf.workers.rollout.hf.huggingface_worker as h; import robotwin.envs.vector_env as v; print(r.__file__); print(h.__file__); print(v.__file__)' \
    >"$run_dir/import_sources.txt" 2>&1
  grep -Fqx "$INSTRUMENTED_ROOT/rlinf/envs/robotwin/robotwin_env.py" "$run_dir/import_sources.txt"
  grep -Fqx "$INSTRUMENTED_ROOT/rlinf/workers/rollout/hf/huggingface_worker.py" "$run_dir/import_sources.txt"
  grep -Fqx "$INSTRUMENTED_ROOT/robotwin/envs/vector_env.py" "$run_dir/import_sources.txt"

  REPO_PATH="$RLINF_ROOT" \
  EMBODIED_PATH="$RLINF_ROOT/examples/embodiment" \
  PYTHONPATH="$INSTRUMENTED_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$ROBOTWIN_ROOT:$RLINF_ROOT" \
  PYTHONHASHSEED=0 \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  XDG_RUNTIME_DIR=/tmp/xdg-robotwin \
  RAY_DEDUP_LOGS=0 \
  "$PYTHON_BIN" "$RLINF_ROOT/examples/embodiment/eval_embodied_agent.py" \
    --config-path "$EVAL_ROOT/runtime_config_tree" \
    --config-name robotwin_adjust_bottle_ppo_openpi_pi05_eval \
    runner.logger.log_path="$run_dir" \
    runner.logger.experiment_name="dev-$variant-batch-$batch" \
    env.eval.total_num_envs="$total" \
    env.eval.assets_path="$ROBOTWIN_ROOT" \
    env.eval.seeds_path="$seed_path" \
    env.eval.video_cfg.save_video=false \
    actor.model.model_path="$MODEL_ROOT" \
    actor.model.num_action_chunks=8 \
    actor.model.action_dim=14 \
    actor.model.add_value_head=true \
    actor.model.openpi.config_name=pi05_aloha_robotwin_head_adjust_bottle \
    actor.model.openpi.num_images_in_input=1 \
    actor.model.openpi.action_chunk=8 \
    actor.model.openpi.action_env_dim=14 \
    actor.model.openpi.noise_level=0.3 \
    runner.ckpt_path="$checkpoint" >"$log_path" 2>&1

  printf 'RESULT variant=%s batch=%s\n' "$variant" "$batch"
  grep -aE "\[INFO .*RLinf\] \{'eval/|OutOfMemory|CUDA out|Exception occurred" \
    "$log_path" | tail -n 5
}

mapfile -t batch_specs < <(
  "$PYTHON_BIN" - <<'PY' "$DEV_ROOT/manifest.json"
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for batch in manifest["batches"]:
    print(f'{int(batch["batch"]):02d}:{int(batch["count"])}')
PY
)
if [[ "${#batch_specs[@]}" -eq 0 ]]; then
  printf 'No preregistered development batches found in %s\n' "$DEV_ROOT" >&2
  exit 5
fi
for spec in "${batch_specs[@]}"; do
  batch="${spec%%:*}"
  total="${spec##*:}"
  if [[ -n "$BATCH_FILTER" && "$batch" != "$BATCH_FILTER" ]]; then
    continue
  fi
  if [[ "$SKIP_BASELINE" != "true" ]]; then
    run_eval baseline "$batch" "$total" null
  fi
  if [[ "$BASELINE_ONLY" != "true" ]]; then
    run_eval "$CANDIDATE_VARIANT" "$batch" "$total" "$CANDIDATE_CHECKPOINT"
  fi
done

printf 'ALL_DEVELOPMENT_EVALUATIONS_COMPLETE candidate=%s baseline_only=%s\n' "$CANDIDATE_VARIANT" "$BASELINE_ONLY"
