#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
ACTOR_SEED="${TRACK2_ACTOR_SEED:-1238}"
PARENT_RELEASE="${TRACK2_PARENT_RELEASE:?TRACK2_PARENT_RELEASE is required}"
PARENT_SCREEN="${TRACK2_PARENT_SCREEN:?TRACK2_PARENT_SCREEN is required}"
MODEL_VERSION="${TRACK2_MODEL_VERSION:?TRACK2_MODEL_VERSION is required}"
BEARER_TOKEN="${TRACK2_BEARER_TOKEN:-local-dev-token}"
AUDIT_ROOT="$ROOT/artifacts/strict_track2_official_20260810"
RLINF_ROOT="$AUDIT_ROOT/full_budget_gpu_cache_release_runtime"
OPENPI_ROOT="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH_ROOT="$AUDIT_ROOT/official_deps/diffsynth_2a2e05f"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD_CKPT="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5_CKPT="$ROOT/artifacts/official_resources/reward_model/t5-base"
V15_ADAPTER="$AUDIT_ROOT/v15_frozen_input_adapter"
CONFIG="$RLINF_ROOT/examples/embodiment/config/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05.yaml"
ENV_CONFIG="$RLINF_ROOT/examples/embodiment/config/env/wan_robotwin_adjust_bottle_http_full.yaml"
BASE_CONTRACT="$ROOT/pipeline/config/strict_track2_full_official_budget_contract.json"
RESOURCE_ADDENDUM="$ROOT/pipeline/config/strict_track2_full_budget_resource_addendum.json"
RUN_ROOT="${TRACK2_RUN_ROOT:-$AUDIT_ROOT/runs/formal_full_budget_${MODEL_VERSION}_seed${ACTOR_SEED}}"
PYTHON_BIN=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
RUNTIME_PYTHONPATH="$DIFFSYNTH_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$RLINF_ROOT"
EXPERIMENT_NAME="wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
CHECKPOINT_ROOT="$RUN_ROOT/$EXPERIMENT_NAME/checkpoints"
CHECKPOINT_KEEP="${TRACK2_CHECKPOINT_KEEP:-2}"

if [[ ! "$CHECKPOINT_KEEP" =~ ^[2-9][0-9]*$ ]]; then
  printf 'TRACK2_CHECKPOINT_KEEP must be an integer >= 2\n' >&2
  exit 2
fi

test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "2f0398e8e6c9be1a32cfe85e69519bb168e2b7edd09c4748eb7ebc8e87719363"
test "$(sha256sum "$ENV_CONFIG" | cut -d' ' -f1)" = "d0992e8e6563311f6e26e83328340eb5a8b842594c548f48c0702ff9eba4a1b5"
test "$(sha256sum "$OPENPI_CKPT/model.safetensors" | cut -d' ' -f1)" = "e8930fd26001d284d22045ed8aeeae95290d278da3f35b42ba398e15e8d74fd6"
test "$(sha256sum "$REWARD_CKPT" | cut -d' ' -f1)" = "5a2066eec31e991a2c006870f73cfc5c3e0afe5b76e9bd91119c467cb9f8de47"
"$PYTHON_BIN" - <<'PY' "$PARENT_SCREEN" "$BASE_CONTRACT" "$RESOURCE_ADDENDUM" "$PARENT_RELEASE"
import json, pathlib, sys
screen, contract, addendum = (json.load(open(path)) for path in sys.argv[1:4])
assert screen["passed"] is True
assert contract["frozen_public_reference_budget"]["max_epochs"] == 1000
assert contract["budget_provenance"]["classification"] == "official-public-full-reference-budget"
assert contract["budget_provenance"]["organizer_hidden_evaluation_budget_claimed_equal"] is False
assert contract["formal_rule"].startswith("The local reference checkpoint")
assert addendum["still_frozen"]["max_epochs"] == 1000
assert addendum["allowed_resource_only_settings"]["actor.enable_offload"] is True
assert addendum["allowed_resource_only_settings"]["rollout.enable_offload"] is True
assert contract["frozen_public_reference_budget"]["save_interval"] == 10
assert addendum["allowed_resource_only_settings"]["service.retrieval_source_cache"] is True
assert addendum["allowed_resource_only_settings"]["service.retrieval_target_cache"] == "lazy hot-candidate cache"
assert addendum["allowed_resource_only_settings"]["service.v141_dead_route_pruning"] is True
assert addendum["allowed_resource_only_settings"]["service_and_bridge.png_codec_workers"] == 8
assert pathlib.Path(sys.argv[4]).is_dir()
PY
capabilities=$(curl -fsS -H "Authorization: Bearer $BEARER_TOKEN" http://127.0.0.1:8001/v1/capabilities)
"$PYTHON_BIN" - <<'PY' "$capabilities" "$MODEL_VERSION"
import json, sys
capabilities = json.loads(sys.argv[1])
assert capabilities["model_version"] == sys.argv[2]
assert capabilities["limits"]["max_batch_size"] == 8
PY
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
mkdir -p "$RUN_ROOT/audit"
"$PYTHON_BIN" - <<'PY' "$RUN_ROOT/audit/effective_checkpoint_resource_overrides.json"
import json, pathlib, sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "format": "strict-track2-resource-only-checkpoint-overrides-v1",
    "official_source_save_interval": 10,
    "effective_save_interval": 1,
    "env_train_enable_offload": True,
    "semantic_effect": "none",
    "reason": (
        "Save every completed policy step as requested; offload the unchanged "
        "HTTP environment reward tensors before FSDP state-dict materialization "
        "to avoid checkpoint-only CUDA OOM. Pi0.5, reward, GRPO, optimizer and "
        "the 1000-step sample budget are unchanged."
    ),
}, indent=2) + "\n")
PY
if [[ ! -e "$RUN_ROOT/audit/formal_run_initialized" ]]; then
  cp "$PARENT_SCREEN" "$RUN_ROOT/audit/selected_parent_screen.json"
  cp "$BASE_CONTRACT" "$RUN_ROOT/audit/"
  cp "$RESOURCE_ADDENDUM" "$RUN_ROOT/audit/"
  cp "$AUDIT_ROOT/full_budget_composed_config.yaml" "$RUN_ROOT/audit/"
  printf '%s\n' "$PARENT_RELEASE" > "$RUN_ROOT/audit/frozen_parent_release.txt"
  printf '%s\n' "$(date -Iseconds)" > "$RUN_ROOT/audit/formal_run_initialized"
else
  cmp -s "$PARENT_SCREEN" "$RUN_ROOT/audit/selected_parent_screen.json"
  cmp -s "$BASE_CONTRACT" "$RUN_ROOT/audit/$(basename "$BASE_CONTRACT")"
  cmp -s "$RESOURCE_ADDENDUM" "$RUN_ROOT/audit/$(basename "$RESOURCE_ADDENDUM")"
  test "$(cat "$RUN_ROOT/audit/frozen_parent_release.txt")" = "$PARENT_RELEASE"
fi

# RLinf's official runner stores complete distributed state under global_step_N
# and natively accepts runner.resume_dir.  Selecting only checkpoints with both
# DCP metadata and exported full weights makes a power-cycle restart safe while
# leaving the optimizer, RNG, algorithm and 1000-step budget untouched.
resume_args=()
latest_step=0
latest_checkpoint=""
if [[ -d "$CHECKPOINT_ROOT" ]]; then
  while IFS= read -r checkpoint; do
    step="${checkpoint##*global_step_}"
    if [[ "$step" =~ ^[0-9]+$ ]] \
      && [[ -s "$checkpoint/actor/dcp_checkpoint/.metadata" ]] \
      && [[ -s "$checkpoint/actor/model_state_dict/full_weights.pt" ]] \
      && (( step > latest_step )); then
      latest_step="$step"
      latest_checkpoint="$checkpoint"
    fi
  done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)
fi
# A power loss can leave the next save directory partially written.  It is not
# resumable and may make RLinf's later save to the same step fail.  Remove only
# numeric, in-scope directories newer than the latest complete recovery point;
# every removal is recorded and no complete checkpoint is touched here.
if [[ -d "$CHECKPOINT_ROOT" ]]; then
  while IFS= read -r checkpoint; do
    step="${checkpoint##*global_step_}"
    if [[ ! "$step" =~ ^[0-9]+$ ]]; then
      printf '%s refusing_non_numeric_partial_checkpoint=%s\n' \
        "$(date -Iseconds)" "$checkpoint" >> "$RUN_ROOT/audit/checkpoint_retention.log"
      exit 9
    fi
    if [[ ! -s "$checkpoint/actor/dcp_checkpoint/.metadata" \
      || ! -s "$checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
      if (( step <= latest_step )); then
        printf '%s refusing_partial_checkpoint_not_newer_than_resume=%s\n' \
          "$(date -Iseconds)" "$checkpoint" >> "$RUN_ROOT/audit/checkpoint_retention.log"
        exit 9
      fi
      case "$checkpoint" in
        "$CHECKPOINT_ROOT"/global_step_[0-9]*)
          printf '%s pruning_incomplete_powerloss_checkpoint=%s resume_step=%s\n' \
            "$(date -Iseconds)" "$checkpoint" "$latest_step" \
            >> "$RUN_ROOT/audit/checkpoint_retention.log"
          rm -rf -- "$checkpoint"
          ;;
        *)
          printf '%s refusing_out_of_scope_partial_checkpoint=%s\n' \
            "$(date -Iseconds)" "$checkpoint" >> "$RUN_ROOT/audit/checkpoint_retention.log"
          exit 9
          ;;
      esac
    fi
  done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)
fi
if (( latest_step >= 1000 )); then
  printf 'FORMAL_FULL_BUDGET_ALREADY_COMPLETE checkpoint=%s\n' "$latest_checkpoint"
  exit 0
fi
if [[ -n "$latest_checkpoint" ]]; then
  resume_args+=("runner.resume_dir=$latest_checkpoint")
  printf '%s resume_from_step=%s checkpoint=%s\n' "$(date -Iseconds)" "$latest_step" "$latest_checkpoint" \
    >> "$RUN_ROOT/audit/training_attempts.log"
else
  printf '%s start_from_step=0\n' "$(date -Iseconds)" >> "$RUN_ROOT/audit/training_attempts.log"
fi

# A checkpoint is roughly 20 GB (DCP recovery state plus exported weights),
# while this server currently has far less than the ~2 TB needed for all 100
# official save points.  This external retention worker never touches an
# incomplete save and keeps the newest two complete recovery points.  Its
# deletion scope is structurally restricted to this run's checkpoint root.
training_active="$RUN_ROOT/audit/training_active"
pruner_pid=""
cleanup_training() {
  rm -f -- "$training_active"
  if [[ -n "$pruner_pid" ]]; then
    kill "$pruner_pid" 2>/dev/null || true
    wait "$pruner_pid" 2>/dev/null || true
  fi
}
trap cleanup_training EXIT

prune_complete_checkpoints() {
  local -a complete=()
  local checkpoint delete_count index step
  while [[ -e "$training_active" ]]; do
    complete=()
    if [[ -d "$CHECKPOINT_ROOT" ]]; then
      while IFS= read -r checkpoint; do
        if [[ -s "$checkpoint/actor/dcp_checkpoint/.metadata" ]] \
          && [[ -s "$checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
          complete+=("$checkpoint")
        fi
      done < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)
    fi
    delete_count=$((${#complete[@]} - CHECKPOINT_KEEP))
    if (( delete_count > 0 )); then
      for ((index = 0; index < delete_count; index++)); do
        checkpoint="${complete[$index]}"
        step="${checkpoint##*global_step_}"
        if [[ ! "$step" =~ ^[0-9]+$ ]]; then
          printf '%s refusing_non_numeric_checkpoint=%s\n' \
            "$(date -Iseconds)" "$checkpoint" \
            >> "$RUN_ROOT/audit/checkpoint_retention.log"
          return 9
        fi
        case "$checkpoint" in
          "$CHECKPOINT_ROOT"/global_step_[0-9]*)
            printf '%s pruning_complete_checkpoint=%s keep_latest=%s\n' \
              "$(date -Iseconds)" "$checkpoint" "$CHECKPOINT_KEEP" \
              >> "$RUN_ROOT/audit/checkpoint_retention.log"
            rm -rf -- "$checkpoint"
            ;;
          *)
            printf '%s refusing_out_of_scope_checkpoint=%s\n' \
              "$(date -Iseconds)" "$checkpoint" \
              >> "$RUN_ROOT/audit/checkpoint_retention.log"
            return 9
            ;;
        esac
      done
    fi
    sleep 60
  done
}

touch "$training_active"
prune_complete_checkpoints &
pruner_pid=$!

"$PYTHON_BIN" -m ray.scripts.scripts stop --force > "$RUN_ROOT/ray_stop_before.log" 2>&1 || true
cd "$RLINF_ROOT"
REPO_PATH="$RLINF_ROOT" \
EMBODIED_PATH="$RLINF_ROOT/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$V15_ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD_CKPT" \
T5_MODEL_PATH="$T5_CKPT" \
PYTHONPATH="$RUNTIME_PYTHONPATH" \
PYTHONHASHSEED=0 \
RAY_DEDUP_LOGS=0 \
"$PYTHON_BIN" "$RLINF_ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF_ROOT/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN_ROOT" \
  runner.save_interval=1 \
  actor.seed="$ACTOR_SEED" \
  actor.enable_offload=true \
  rollout.enable_offload=true \
  env.train.enable_offload=true \
  +weight_syncer.patch.transport_device=cpu \
  "${resume_args[@]}" >> "$RUN_ROOT/launcher.log" 2>&1

cleanup_training
trap - EXIT
printf 'FORMAL_FULL_BUDGET_COMPLETE run=%s\n' "$RUN_ROOT"
