#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_RL_PYTHON:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
BASE="$AUDIT/full_budget_gpu_cache_release_runtime"
RUNTIME="$AUDIT/probability_consistent_four_step_v169_runtime"
RUN="$AUDIT/runs/probability_consistent_four_step_v169_seed1243_retry2"
PATCH="$ROOT/pipeline/patches/rlinf_openpi_frozen_vlm_mode_consistency.patch"
PREREG="$ROOT/pipeline/config/strict_track2_v169_probability_consistent_one_update_preregistration.json"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$AUDIT/official_deps/diffsynth_2a2e05f"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
ADAPTER="$AUDIT/v15_frozen_input_adapter"
RELEASE="$JOINT/v169_instruction_arm_routed_release"
SERVICE_DIR="$AUDIT/probability_consistent_service_four_step_v169_retry2"
BRIDGE_DIR="$AUDIT/probability_consistent_bridge_four_step_v169_retry2"
MODEL_VERSION="track2-v16.9-instruction-arm-routed"

test "$(sha256sum "$BASE/rlinf/workers/actor/fsdp_actor_worker.py" | cut -d' ' -f1)" = "24347c8d08e99fc273630f874ee5435e934a8fdfb028f0fb1c937c258edaeefb"
test -s "$RELEASE/cache_replay_report.json"
test "$("$GO1" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "$RELEASE/cache_replay_report.json")" = 1
test -s "$AUDIT/service_acceptance/v169_protocol_acceptance.json"
test "$("$GO1" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["passed"]))' "$AUDIT/service_acceptance/v169_protocol_acceptance.json")" = 1
test -s "$RUNTIME/rlinf/workers/actor/fsdp_actor_worker.py"
test ! -e "$RUN"; test ! -e "$SERVICE_DIR"; test ! -e "$BRIDGE_DIR"
mkdir -p "$RUN/audit" "$SERVICE_DIR" "$BRIDGE_DIR"
cp "$PREREG" "$RUN/audit/"
sha256sum "$OPENPI_CKPT/model.safetensors" "$OPENPI_CKPT/metadata.pt" \
  "$OPENPI_CKPT/rlinf/robotwin_headcam_adjust_bottle/norm_stats.json" \
  "$REWARD" "$RELEASE/v169_arm_routed_manifest.json" > "$RUN/audit/frozen_inputs_sha256.txt"
"$PY" -m py_compile "$RUNTIME/rlinf/workers/actor/fsdp_actor_worker.py"

service_pid=""; bridge_pid=""
cleanup() {
  if [[ -n "$bridge_pid" ]]; then kill "$bridge_pid" 2>/dev/null || true; fi
  if [[ -n "$service_pid" ]]; then kill "$service_pid" 2>/dev/null || true; fi
  "$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_after.log" 2>&1 || true
}
trap cleanup EXIT

cd "$ROOT"
PYTHONPATH="$ROOT/pipeline" WAM_MODEL_VERSION="$MODEL_VERSION" WAM_BEARER_TOKEN="local-dev-token" \
WAM_BACKEND="v169-arm-routed" WAM_CHECKPOINT_DIR="$RELEASE" WAM_V15_LIBRARY_DIR="$ROOT/artifacts" \
WAM_DEVICE="cuda" WAM_PORT=8001 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=8 \
WAM_RELEASE_CUDA_CACHE=1 WAM_RETRIEVAL_SOURCE_CACHE=1 \
"$GO1" pipeline/scripts/serve.py > "$SERVICE_DIR/service.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8001/v1/health >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:8001/v1/health | grep -q "$MODEL_VERSION"

PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server \
  --world-model-url http://127.0.0.1:8001 --token local-dev-token --model-version "$MODEL_VERSION" \
  --host 127.0.0.1 --port 18080 --audit-max-items 1 > "$BRIDGE_DIR/bridge.log" 2>&1 &
bridge_pid=$!
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'

"$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_before.log" 2>&1 || true
cd "$RUNTIME"
REPO_PATH="$RUNTIME" EMBODIED_PATH="$RUNTIME/examples/embodiment" OPENPI_CKPT_PATH="$OPENPI_CKPT" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RUNTIME" PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 \
"$PY" "$RUNTIME/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RUNTIME/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.max_steps=4 runner.max_epochs=1000 runner.save_interval=1 \
  actor.seed=1243 actor.enable_offload=true rollout.enable_offload=true \
  +weight_syncer.patch.transport_device=cpu > "$RUN/launcher.log" 2>&1

if grep -aqE "CUDA out of memory|OutOfMemoryError|HTTP.*(500|502|503|504)" "$RUN/launcher.log"; then exit 7; fi
test -s "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
printf 'V169_PROBABILITY_CONSISTENT_FOUR_STEP_RETRY2_COMPLETE_NONFORMAL\n'
