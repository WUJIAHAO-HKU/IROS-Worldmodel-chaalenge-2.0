#!/usr/bin/env bash
set -u

RLINF_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
ROBOTWIN_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
OPENPI_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/openpi-rlinf-full"
AUDIT_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810"
EVAL_ROOT="$AUDIT_ROOT/real_robotwin_eval"
INSTRUMENTED_ROOT="${TRACK2_INSTRUMENTED_RUNTIME_ROOT:-$EVAL_ROOT/instrumented_runtime}"
OUTPUT_ROOT="${TRACK2_INSTRUMENTED_OUTPUT_ROOT:-$EVAL_ROOT/instrumented_metrics}"
MODEL_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/pi05_adjust_bottle"
CHECKPOINT="${TRACK2_FINAL128_CHECKPOINT:?set TRACK2_FINAL128_CHECKPOINT}"
CANDIDATE_VARIANT="${TRACK2_FINAL128_VARIANT:?set TRACK2_FINAL128_VARIANT}"
PYTHON_BIN="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
BATCHES="${TRACK2_INSTRUMENTED_BATCHES:-00 01 02 03 04 05 06 07 08}"
RUN_BASELINE="${TRACK2_FINAL128_RUN_BASELINE:-true}"

if [[ "$RUN_BASELINE" != "true" && "$RUN_BASELINE" != "false" ]]; then
  printf 'TRACK2_FINAL128_RUN_BASELINE must be true or false\n' >&2
  exit 92
fi

run_eval() {
  local variant="$1"
  local batch="$2"
  local total="$3"
  local checkpoint="$4"
  local run_dir="$OUTPUT_ROOT/$variant/batch16_$batch"
  local log_path="$run_dir/launcher.log"
  local seed_path="$EVAL_ROOT/seed_batches16/batch_$batch.json"

  if [[ "$batch" == "00" ]]; then
    seed_path="$EVAL_ROOT/seed_shards/shard_00.json"
  fi

  if grep -aq "'eval/num_trajectories': $total" "$log_path" 2>/dev/null \
    && grep -aq "'eval/grasp_once'" "$log_path" 2>/dev/null \
    && grep -aq "'eval/success_bitmask'" "$log_path" 2>/dev/null; then
    printf 'SKIP variant=%s batch=%s reason=complete\n' "$variant" "$batch"
    return 0
  fi

  "$PYTHON_BIN" -m ray.scripts.scripts stop --force \
    >"/tmp/ray_stop_track2_instrumented_${variant}_${batch}.log" 2>&1 || true
  mkdir -p "$run_dir"
  cd /tmp || return 90

  REPO_PATH="$RLINF_ROOT" \
  PYTHONPATH="$INSTRUMENTED_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$ROBOTWIN_ROOT:$RLINF_ROOT" \
  PYTHONHASHSEED=0 \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$PYTHON_BIN" -c \
    'import rlinf.envs.robotwin.robotwin_env as r; import rlinf.workers.rollout.hf.huggingface_worker as h; import robotwin.envs.vector_env as v; print(r.__file__); print(h.__file__); print(v.__file__)' \
    >"$run_dir/import_sources.txt" 2>&1
  if ! grep -Fqx "$INSTRUMENTED_ROOT/rlinf/envs/robotwin/robotwin_env.py" "$run_dir/import_sources.txt" \
    || ! grep -Fqx "$INSTRUMENTED_ROOT/rlinf/workers/rollout/hf/huggingface_worker.py" "$run_dir/import_sources.txt" \
    || ! grep -Fqx "$INSTRUMENTED_ROOT/robotwin/envs/vector_env.py" "$run_dir/import_sources.txt"; then
    printf 'IMPORT_ASSERTION_FAILED variant=%s batch=%s\n' "$variant" "$batch" >&2
    cat "$run_dir/import_sources.txt" >&2
    return 91
  fi

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
    runner.logger.experiment_name="instrumented-$variant-batch16-$batch" \
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
  local exit_code=$?

  printf 'RESULT variant=%s batch=%s exit=%s\n' "$variant" "$batch" "$exit_code"
  grep -aE "\[INFO .*RLinf\] \{'eval/|OutOfMemory|CUDA out|Exception occurred" \
    "$log_path" | tail -n 5
  return "$exit_code"
}

for batch in $BATCHES; do
  total=16
  if [[ "$batch" == "00" || "$batch" == "08" ]]; then
    total=8
  fi
  if [[ "$RUN_BASELINE" == "true" ]]; then
    run_eval baseline "$batch" "$total" null || exit $?
  fi
  run_eval "$CANDIDATE_VARIANT" "$batch" "$total" "$CHECKPOINT" || exit $?
done

printf 'ALL_INSTRUMENTED_EVALUATIONS_COMPLETE candidate=%s\n' "$CANDIDATE_VARIANT"
