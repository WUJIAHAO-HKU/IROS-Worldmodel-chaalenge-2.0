#!/usr/bin/env bash
set -euo pipefail

ROOT="${TRACK2_ROOT:-/root/autodl-tmp/IROS_WAM_2.0 challenge}"
OPENPI_ROOT="$ROOT/third_party/openpi-rlinf-full"
AUDIT="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RLINF="$AUDIT/full_budget_gpu_cache_release_runtime"
DIFFSYNTH="$AUDIT/official_deps/diffsynth_2a2e05f"
CONFIG="$RLINF/examples/embodiment/config/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05.yaml"
ENV_CONFIG="$RLINF/examples/embodiment/config/env/wan_robotwin_adjust_bottle_http_full.yaml"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD_CKPT="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5_CKPT="$ROOT/artifacts/official_resources/reward_model/t5-base"
ADAPTER="$AUDIT/v15_frozen_input_adapter"
CANDIDATE="$J/v166_singlepass_student_pilot_seed166/best"
SCREEN="$J/v166_singlepass_student_pilot_seed166/screen/v166_offline_screen.json"
MIXED_SCREEN="$J/v166_singlepass_student_pilot_seed166/screen/v166_mixed_visual_screen.json"
PREREG="$ROOT/pipeline/config/strict_track2_v166_one_update_preregistration.json"
RUN="$AUDIT/runs/full_budget_one_update_v166_singlepass_seed1238"
BRIDGE_ROOT="$AUDIT/v166_singlepass_one_update_bridge"
MODEL_VERSION="track2-v16.6-singlepass-student-seed166"
RL_PY=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
GO1_PY=/root/miniconda3/envs/go1/bin/python
RUNTIME_PYTHONPATH="$DIFFSYNTH:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$RLINF"

test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "2f0398e8e6c9be1a32cfe85e69519bb168e2b7edd09c4748eb7ebc8e87719363"
test "$(sha256sum "$ENV_CONFIG" | cut -d' ' -f1)" = "d0992e8e6563311f6e26e83328340eb5a8b842594c548f48c0702ff9eba4a1b5"
test "$(sha256sum "$OPENPI_CKPT/model.safetensors" | cut -d' ' -f1)" = "e8930fd26001d284d22045ed8aeeae95290d278da3f35b42ba398e15e8d74fd6"
test "$(sha256sum "$REWARD_CKPT" | cut -d' ' -f1)" = "5a2066eec31e991a2c006870f73cfc5c3e0afe5b76e9bd91119c467cb9f8de47"
"$GO1_PY" - "$SCREEN" "$MIXED_SCREEN" "$PREREG" "$CANDIDATE" "$ROOT" <<'PY'
import hashlib, json, pathlib, sys
screen=json.load(open(sys.argv[1])); mixed=json.load(open(sys.argv[2])); prereg=json.load(open(sys.argv[3])); candidate=pathlib.Path(sys.argv[4]).resolve(); root=pathlib.Path(sys.argv[5])
assert screen['passed'] is True
assert mixed['passed'] is True
assert pathlib.Path(screen['candidate']).resolve() == candidate
assert pathlib.Path(mixed['candidate']).resolve() == candidate
assert prereg['format']=='strict-track2-v166-one-update-preregistration-v1'
assert prereg['classification'].startswith('resource and policy-signal diagnostic')
assert prereg['official_training']['runner.max_steps']==1
assert prereg['official_training']['actor.seed']==1238
assert hashlib.sha256((root/prereg['runner']['script']).read_bytes()).hexdigest()==prereg['runner']['script_sha256']
PY
test ! -e "$RUN"
test ! -e "$BRIDGE_ROOT"
if ss -ltn | grep -Eq ':8001 |:18080 '; then
  echo "ports 8001 or 18080 are already occupied" >&2
  exit 3
fi

mkdir -p "$RUN/audit" "$BRIDGE_ROOT/bridge_audit"
cp "$PREREG" "$RUN/audit/"
cp "$SCREEN" "$RUN/audit/selected_parent_screen.json"
cp "$MIXED_SCREEN" "$RUN/audit/selected_parent_mixed_visual_screen.json"
cp "$ROOT/pipeline/config/strict_track2_full_official_budget_contract.json" "$RUN/audit/"
cp "$ROOT/pipeline/config/strict_track2_full_budget_resource_addendum.json" "$RUN/audit/"

service_pid=""
bridge_pid=""
sampler_pid=""
cleanup() {
  for pid in "$sampler_pid" "$bridge_pid" "$service_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  "$RL_PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_after.log" 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "$ROOT"
env PYTHONPATH="$ROOT/pipeline" \
  WAM_BACKEND=multisource-flow-unet \
  WAM_BEARER_TOKEN=local-dev-token \
  WAM_CHECKPOINT_DIR="$CANDIDATE" \
  WAM_DEVICE=cuda \
  WAM_MODEL_VERSION="$MODEL_VERSION" \
  WAM_PORT=8001 \
  WAM_BATCH_WORKERS=1 \
  WAM_NATIVE_BATCH_ENABLED=1 \
  WAM_NATIVE_BATCH_MICRO_SIZE=8 \
  WAM_RELEASE_CUDA_CACHE=1 \
  "$GO1_PY" pipeline/scripts/serve.py > "$RUN/service.log" 2>&1 &
service_pid=$!
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8001/v1/health | grep -q "$MODEL_VERSION"; then break; fi
  sleep 1
done
capabilities=$(curl -fsS -H 'Authorization: Bearer local-dev-token' http://127.0.0.1:8001/v1/capabilities)
"$GO1_PY" - "$capabilities" "$MODEL_VERSION" <<'PY'
import json,sys
j=json.loads(sys.argv[1]); assert j['model_version']==sys.argv[2]; assert j['limits']['max_batch_size']==8
PY

env PYTHONPATH="$ROOT/pipeline" TRACK2_BRIDGE_AUDIT_MAX_ITEMS=1 \
  "$GO1_PY" -m wam_pipeline.rlinf_bridge.server \
  --world-model-url http://127.0.0.1:8001 \
  --token local-dev-token \
  --model-version "$MODEL_VERSION" \
  --port 18080 \
  --timeout 900 \
  --audit-dir "$BRIDGE_ROOT/bridge_audit" \
  > "$BRIDGE_ROOT/bridge.log" 2>&1 &
bridge_pid=$!
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:18080/health | grep -q ready; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:18080/health | grep -q ready

nvidia-smi --query-compute-apps=timestamp,pid,used_memory,name --format=csv,noheader -lms 500 \
  > "$RUN/audit/gpu_process_memory_500ms.csv" &
sampler_pid=$!
printf '%s\n' "$service_pid" > "$RUN/audit/service.pid"
printf '%s\n' "$bridge_pid" > "$RUN/audit/bridge.pid"
printf '%s\n' "$sampler_pid" > "$RUN/audit/gpu_sampler.pid"

"$RL_PY" -m ray.scripts.scripts stop --force > "$RUN/ray_stop_before.log" 2>&1 || true
cd "$RLINF"
REPO_PATH="$RLINF" \
EMBODIED_PATH="$RLINF/examples/embodiment" \
OPENPI_CKPT_PATH="$OPENPI_CKPT" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ADAPTER" \
ROBOTWIN_REWARD_MODEL_PATH="$REWARD_CKPT" \
T5_MODEL_PATH="$T5_CKPT" \
PYTHONPATH="$RUNTIME_PYTHONPATH" \
PYTHONHASHSEED=0 \
RAY_DEDUP_LOGS=0 \
"$RL_PY" "$RLINF/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" \
  runner.max_steps=1 \
  actor.seed=1238 \
  actor.enable_offload=true \
  rollout.enable_offload=true \
  +weight_syncer.patch.transport_device=cpu > "$RUN/launcher.log" 2>&1

nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader > "$RUN/audit/gpu_after_completion.csv"
printf 'V166_ONE_UPDATE_COMPLETE_NONFORMAL run=%s\n' "$RUN"
