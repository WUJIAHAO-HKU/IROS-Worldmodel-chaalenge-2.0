#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
PY="${TRACK2_RL_PYTHON:-/root/autodl-tmp/conda_envs/rlinf_track2/bin/python}"
GO1="${TRACK2_GO1_PYTHON:-/root/miniconda3/envs/go1/bin/python}"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
BASE="$AUDIT/full_budget_gpu_cache_release_runtime"
RUNTIME="$AUDIT/preupdate_logprob_audit_runtime_v5"
RUN="$AUDIT/runs/preupdate_logprob_consistency_v157_seed1242_v5"
PREREG="$ROOT/pipeline/config/strict_track2_preupdate_logprob_consistency_preregistration.json"
PATCH="$ROOT/pipeline/patches/rlinf_preupdate_logprob_observation.patch"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFFSYNTH="$AUDIT/official_deps/diffsynth_2a2e05f"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
ADAPTER="$AUDIT/v15_frozen_input_adapter"
RELEASE="$JOINT/v157_hybrid_gate_blend12_formal_release"
SERVICE_DIR="$AUDIT/preupdate_logprob_service_v5"
BRIDGE_DIR="$AUDIT/preupdate_logprob_bridge_v5"
MODEL_VERSION="track2-v15.7-hybrid-action-gated-reward-safe-blend12"
ACTOR="$BASE/rlinf/workers/actor/fsdp_actor_worker.py"

test "$(sha256sum "$ACTOR" | cut -d' ' -f1)" = "24347c8d08e99fc273630f874ee5435e934a8fdfb028f0fb1c937c258edaeefb"
test ! -e "$RUNTIME"
test ! -e "$RUN"
test ! -e "$SERVICE_DIR"
test ! -e "$BRIDGE_DIR"
mkdir -p "$RUN/audit" "$SERVICE_DIR" "$BRIDGE_DIR"
cp "$PREREG" "$RUN/audit/"
cp -a "$BASE" "$RUNTIME"
patch --batch --forward -p1 -d "$RUNTIME" < "$PATCH"
"$PY" -m py_compile "$RUNTIME/rlinf/workers/actor/fsdp_actor_worker.py"

service_pid=""
bridge_pid=""
cleanup() {
  if [[ -n "$bridge_pid" ]]; then kill "$bridge_pid" 2>/dev/null || true; fi
  if [[ -n "$service_pid" ]]; then kill "$service_pid" 2>/dev/null || true; fi
  "$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_after.log" 2>&1 || true
}
trap cleanup EXIT

cd "$ROOT"
PYTHONPATH="$ROOT/pipeline" \
WAM_MODEL_VERSION="$MODEL_VERSION" \
WAM_BEARER_TOKEN="local-dev-token" \
WAM_BACKEND="v15-gated-composite" \
WAM_CHECKPOINT_DIR="$RELEASE" \
WAM_V15_LIBRARY_DIR="$ROOT/artifacts" \
WAM_DEVICE="cuda" \
WAM_PORT=8001 \
WAM_NATIVE_BATCH_ENABLED=1 \
WAM_NATIVE_BATCH_MICRO_SIZE=8 \
WAM_RELEASE_CUDA_CACHE=1 \
"$GO1" pipeline/scripts/serve.py > "$SERVICE_DIR/service.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/v1/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8001/v1/health | grep -q "$MODEL_VERSION"

PYTHONPATH="$ROOT/pipeline" "$GO1" -m wam_pipeline.rlinf_bridge.server \
  --world-model-url http://127.0.0.1:8001 \
  --token local-dev-token --model-version "$MODEL_VERSION" \
  --host 127.0.0.1 --port 18080 --audit-max-items 0 \
  > "$BRIDGE_DIR/bridge.log" 2>&1 &
bridge_pid=$!
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:18080/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'

"$PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_before.log" 2>&1 || true
set +e
cd "$RUNTIME"
REPO_PATH="$RUNTIME" EMBODIED_PATH="$RUNTIME/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD" T5_MODEL_PATH="$T5" \
PYTHONPATH="$DIFFSYNTH:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RUNTIME" \
PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 TRACK2_ABORT_AFTER_PREUPDATE_LOGPROB_AUDIT=1 \
"$PY" "$RUNTIME/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RUNTIME/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.max_steps=1 runner.max_epochs=1 \
  runner.save_interval=1000 algorithm.rollout_epoch=1 \
  env.train.total_num_envs=4 env.train.group_size=4 \
  env.train.max_episode_steps=8 env.train.max_steps_per_rollout_epoch=8 \
  actor.global_batch_size=4 actor.micro_batch_size=4 actor.seed=1242 \
  actor.enable_offload=true rollout.enable_offload=true \
  +weight_syncer.patch.transport_device=cpu > "$RUN/launcher.log" 2>&1
status=$?
set -e
grep -a "TRACK2_PREUPDATE_LOGPROB_AUDIT" "$RUN/launcher.log" | tee "$RUN/audit/preupdate_logprob_metric.log"
grep -aq "TRACK2_EXPECTED_ABORT_AFTER_PREUPDATE_LOGPROB_AUDIT" "$RUN/launcher.log"
if grep -aqE "CUDA out of memory|OutOfMemoryError" "$RUN/launcher.log"; then
  echo "unexpected CUDA OOM in logprob audit" >&2
  exit 7
fi
if [[ "$status" -eq 0 ]]; then
  echo "diagnostic unexpectedly completed instead of aborting before training" >&2
  exit 8
fi
printf 'PREUPDATE_LOGPROB_DIAGNOSTIC_COMPLETE_EXPECTED_ABORT\n'
