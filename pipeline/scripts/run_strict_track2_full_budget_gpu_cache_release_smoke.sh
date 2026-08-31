#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge"
OFFICIAL_RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI_ROOT="$ROOT/third_party/openpi-rlinf-full"
AUDIT_ROOT="$ROOT/artifacts/strict_track2_official_20260810"
JOINT_ROOT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
RLINF_ROOT="$AUDIT_ROOT/full_budget_gpu_cache_release_runtime"
DIFFSYNTH_ROOT="$AUDIT_ROOT/official_deps/diffsynth_2a2e05f"
CONFIG="$RLINF_ROOT/examples/embodiment/config/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05.yaml"
ENV_CONFIG="$RLINF_ROOT/examples/embodiment/config/env/wan_robotwin_adjust_bottle_http_full.yaml"
OPENPI_CKPT="$ROOT/artifacts/official_resources/pi05_adjust_bottle"
REWARD_CKPT="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5_CKPT="$ROOT/artifacts/official_resources/reward_model/t5-base"
V15_ADAPTER="$AUDIT_ROOT/v15_frozen_input_adapter"
PARENT_SCREEN="$JOINT_ROOT/v157_reward_safe_parent_screen.json"
PREREG="$ROOT/pipeline/config/strict_track2_v157_gpu_cache_release_preregistration.json"
RUN_ROOT="$AUDIT_ROOT/runs/full_budget_gpu_cache_release_v157_seed1238"
PYTHON_BIN=/root/autodl-tmp/conda_envs/rlinf_track2/bin/python
RUNTIME_PYTHONPATH="$DIFFSYNTH_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$RLINF_ROOT"

test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "2f0398e8e6c9be1a32cfe85e69519bb168e2b7edd09c4748eb7ebc8e87719363"
test "$(sha256sum "$ENV_CONFIG" | cut -d' ' -f1)" = "d0992e8e6563311f6e26e83328340eb5a8b842594c548f48c0702ff9eba4a1b5"
test "$(sha256sum "$OPENPI_CKPT/model.safetensors" | cut -d' ' -f1)" = "e8930fd26001d284d22045ed8aeeae95290d278da3f35b42ba398e15e8d74fd6"
test "$(sha256sum "$REWARD_CKPT" | cut -d' ' -f1)" = "5a2066eec31e991a2c006870f73cfc5c3e0afe5b76e9bd91119c467cb9f8de47"
test "$(sha256sum "$RLINF_ROOT/rlinf/workers/actor/fsdp_actor_worker.py" | cut -d' ' -f1)" = "24347c8d08e99fc273630f874ee5435e934a8fdfb028f0fb1c937c258edaeefb"
test "$(sha256sum "$RLINF_ROOT/rlinf/workers/rollout/hf/huggingface_worker.py" | cut -d' ' -f1)" = "0b6948272e53cda6ae18d90946073c417d1f844d7a1a0de4902747d990dbe37d"
test "$(sha256sum "$RLINF_ROOT/rlinf/envs/world_model/world_model_wan_http_env.py" | cut -d' ' -f1)" = "40ef65a82c39eaa0c6c28b1669d3185706c630dc4c1fd30b5e9f09f79e5557ca"
"$PYTHON_BIN" - <<'PY' "$PARENT_SCREEN" "$PREREG"
import json, sys
assert json.load(open(sys.argv[1]))["passed"] is True
registration = json.load(open(sys.argv[2]))
assert registration["classification"].startswith("resource-feasibility diagnostic")
assert registration["official_rl"]["diagnostic_max_steps"] == 1
assert registration["official_rl"]["policy_weights_changed"] is False
assert registration["official_rl"]["reward_changed"] is False
assert registration["official_rl"]["actions_changed"] is False
assert registration["official_rl"]["algorithm_or_formal_budget_changed"] is False
PY
capabilities=$(curl -fsS -H 'Authorization: Bearer local-dev-token' http://127.0.0.1:8001/v1/capabilities)
"$PYTHON_BIN" - <<'PY' "$capabilities"
import json, sys
capabilities = json.loads(sys.argv[1])
assert capabilities["model_version"] == "track2-v15.7-hybrid-action-gated-reward-safe-blend12"
assert capabilities["limits"]["max_batch_size"] == 8
PY
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/audit"
cp "$PREREG" "$RUN_ROOT/audit/"
cp "$PARENT_SCREEN" "$RUN_ROOT/audit/selected_parent_screen.json"
cp "$ROOT/pipeline/config/strict_track2_full_official_budget_contract.json" "$RUN_ROOT/audit/"
cp "$AUDIT_ROOT/full_budget_composed_config.yaml" "$RUN_ROOT/audit/"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader > "$RUN_ROOT/audit/gpu_before_launch.csv"

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
  runner.max_steps=1 \
  actor.seed=1238 \
  +weight_syncer.patch.transport_device=cpu > "$RUN_ROOT/launcher.log" 2>&1

nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader > "$RUN_ROOT/audit/gpu_after_completion.csv"
printf 'RESOURCE_SMOKE_COMPLETE_NONFORMAL\n'
