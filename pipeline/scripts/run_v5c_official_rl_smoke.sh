#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/root/autodl-tmp/IROS_WAM_2.0 challenge}
RUNTIME=/root/autodl-tmp/iros_v15_rl_probability_audit_v2
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI="$ROOT/third_party/openpi-rlinf-full"
DIFF="$ROOT/artifacts/strict_track2_official_20260810/official_deps/diffsynth_2a2e05f"
RELEASE="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/recursive_arm_routed_v5c_alpha020/release"
RUN="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/recursive_arm_routed_v5c_alpha020/rl_smoke"
PY=/root/miniconda3/envs/go1/bin/python
mkdir -p "$RUN"
BRIDGE=""
export PYTHONPATH="$ROOT/pipeline"
WAM_MODEL_VERSION=track2-v5c-alpha020-instruction-routed WAM_BEARER_TOKEN=local-dev-token \
WAM_BACKEND=v15-gated-composite WAM_CHECKPOINT_DIR="$RELEASE" WAM_V15_LIBRARY_DIR="$ROOT/artifacts" \
WAM_DEVICE=cuda WAM_PORT=8001 WAM_NATIVE_BATCH_ENABLED=1 WAM_NATIVE_BATCH_MICRO_SIZE=2 \
"$PY" "$ROOT/pipeline/scripts/serve.py" >"$RUN/service.log" 2>&1 & SERVICE=$!
trap 'kill "$BRIDGE" 2>/dev/null || true; kill "$SERVICE" 2>/dev/null || true; "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8001/v1/health 2>/dev/null | grep -q v5c-alpha020 && break; sleep 1; done
curl -fsS http://127.0.0.1:8001/v1/health | grep -q v5c-alpha020
PYTHONPATH="$ROOT/pipeline" WAM_BEARER_TOKEN=local-dev-token \
"$PY" -m wam_pipeline.rlinf_bridge.server --world-model-url http://127.0.0.1:8001 \
  --token local-dev-token --model-version track2-v5c-alpha020-instruction-routed \
  --host 127.0.0.1 --port 18080 --audit-max-items 1 >"$RUN/bridge.log" 2>&1 & BRIDGE=$!
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18080/health 2>/dev/null | grep -q '"status":"ready"' && break; sleep 1; done
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
cd "$RLINF"
REPO_PATH="$RLINF" EMBODIED_PATH="$RLINF/examples/embodiment" \
OPENPI_CKPT_PATH="$ROOT/artifacts/official_resources/pi05_adjust_bottle" \
WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT="$ROOT/artifacts/strict_track2_official_20260810/v15_frozen_input_adapter" \
ROBOTWIN_REWARD_MODEL_PATH="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
T5_MODEL_PATH="$ROOT/artifacts/official_resources/reward_model/t5-base" \
PYTHONPATH="$DIFF:$OPENPI/packages/openpi-client/src:$OPENPI/src:$RLINF" PYTHONHASHSEED=0 RAY_DEDUP_LOGS=0 \
"$PY" "$RLINF/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN" runner.max_steps=1 runner.max_epochs=1 runner.save_interval=1 \
  env.train.max_steps_per_rollout_epoch=8 env.train.max_episode_steps=8 \
  actor.global_batch_size=256 actor.micro_batch_size=8 \
  actor.seed=2026 actor.enable_offload=true rollout.enable_offload=true \
  +weight_syncer.patch.transport_device=cpu >"$RUN/launcher.log" 2>&1
test -s "$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
echo V5C_OFFICIAL_RL_SMOKE_COMPLETE
