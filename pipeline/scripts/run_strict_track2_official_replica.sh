#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s ACTOR_SEED\n' "$0" >&2
  exit 2
fi
ACTOR_SEED="$1"

RLINF_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
OPENPI_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/third_party/openpi-rlinf-full"
AUDIT_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810"
JOINT_ROOT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810"
DIFFSYNTH_ROOT="$AUDIT_ROOT/official_deps/diffsynth_2a2e05f"
CONFIG="$RLINF_ROOT/examples/embodiment/config/wan_robotwin_adjust_bottle_http_grpo_openpi_pi05.yaml"
ENV_CONFIG="$RLINF_ROOT/examples/embodiment/config/env/wan_robotwin_adjust_bottle_http.yaml"
OPENPI_CKPT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/pi05_adjust_bottle"
REWARD_CKPT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5_CKPT="/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/official_resources/reward_model/t5-base"
V15_ADAPTER="$AUDIT_ROOT/v15_frozen_input_adapter"
PREREGISTRATION="${TRACK2_POLICY_PREREGISTRATION:-$AUDIT_ROOT/audit/strict_track2_replica_preregistration.json}"
PARENT_SCREEN=${TRACK2_PARENT_SCREEN:-$AUDIT_ROOT/../strict_track2_joint_augmentation_20260810/formal_v15_gated_full_composite_screen.json}
PARENT_REWARD_ALIGNMENT=${TRACK2_PARENT_REWARD_ALIGNMENT:-}
JOINT_GOAL="$AUDIT_ROOT/audit/strict_track2_joint_optimization_goal.json"
PARENT_PREREGISTRATION="${TRACK2_PARENT_PREREGISTRATION:-$AUDIT_ROOT/audit/strict_track2_parent_adaptation_preregistration.json}"
RUN_LABEL="${TRACK2_RUN_LABEL:-official_replica_seed_${ACTOR_SEED}}"
if [[ ! "$RUN_LABEL" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  printf 'Invalid TRACK2_RUN_LABEL: %s\n' "$RUN_LABEL" >&2
  exit 2
fi
RUN_ROOT="$AUDIT_ROOT/runs/$RUN_LABEL"
EXPERIMENT="wan_robotwin_adjust_bottle_http_grpo_openpi_pi05"
CHECKPOINT="$RUN_ROOT/$EXPERIMENT/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
PYTHON_BIN="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
RUNTIME_PYTHONPATH="$DIFFSYNTH_ROOT:$OPENPI_ROOT/packages/openpi-client/src:$OPENPI_ROOT/src:$RLINF_ROOT"

case "$ACTOR_SEED" in
  1235|1236|1237|1238|1239|1240|1241) ;;
  *) printf 'Actor seed %s is not preregistered\n' "$ACTOR_SEED" >&2; exit 2 ;;
esac

if pgrep -af 'eval_embodied_agent.py' >/dev/null; then
  printf 'Refusing to interrupt an active real RoboTwin evaluation\n' >&2
  exit 3
fi
if [[ -e "$RUN_ROOT" && ! -f "$CHECKPOINT" ]]; then
  printf 'Partial replica run already exists; inspect it manually: %s\n' "$RUN_ROOT" >&2
  exit 4
fi
if [[ -f "$CHECKPOINT" ]]; then
  printf 'SKIP actor_seed=%s reason=checkpoint_exists path=%s\n' "$ACTOR_SEED" "$CHECKPOINT"
  exit 0
fi

declare -A EXPECTED_HASHES=(
  ["$CONFIG"]="46ddd1906355d81dfb98bf95b182de26d0ce21d3bb41c195152ca0deb5d0df82"
  ["$ENV_CONFIG"]="e430ac2f7ff15bfe27d97f392f9f07f2d8ed28303ef52063650f7ab9af500c3d"
  ["$RLINF_ROOT/rlinf/workers/actor/fsdp_actor_worker.py"]="6ab21707195a314f46a1985391c8457406f6a8fdcfdf8e884bd484d914cb4235"
  ["$RLINF_ROOT/rlinf/envs/world_model/world_model_wan_http_env.py"]="e94064a79bcd14454fd5021532f8605c0dd944c440849fd5fc65967789e7a2b6"
  ["$OPENPI_CKPT/model.safetensors"]="e8930fd26001d284d22045ed8aeeae95290d278da3f35b42ba398e15e8d74fd6"
  ["$REWARD_CKPT"]="5a2066eec31e991a2c006870f73cfc5c3e0afe5b76e9bd91119c467cb9f8de47"
  ["$T5_CKPT/config.json"]="46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033"
  ["$T5_CKPT/model.safetensors"]="a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"
  ["$T5_CKPT/spiece.model"]="d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"
  ["$T5_CKPT/tokenizer.json"]="d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"
)
for path in "${!EXPECTED_HASHES[@]}"; do
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "${EXPECTED_HASHES[$path]}" ]]; then
    printf 'Frozen input hash mismatch: %s expected=%s actual=%s\n' \
      "$path" "${EXPECTED_HASHES[$path]}" "$actual" >&2
    exit 5
  fi
done
if grep -qiE 'mpc|candidate[_ -]?action|action[_ -]?scale' "$CONFIG" "$ENV_CONFIG"; then
  printf 'Forbidden action-selection term found in official training config\n' >&2
  exit 6
fi

curl -fsS http://127.0.0.1:8001/v1/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:18080/health | grep -q '"status":"ready"'
WORLD_MODEL_CAPABILITIES="$(curl -fsS -H 'Authorization: Bearer local-dev-token' http://127.0.0.1:8001/v1/capabilities)"
WORLD_MODEL_HEALTH="$(curl -fsS http://127.0.0.1:8001/v1/health)"
"$PYTHON_BIN" - <<'PY' "$PARENT_SCREEN" "$WORLD_MODEL_HEALTH" "$WORLD_MODEL_CAPABILITIES" "$PREREGISTRATION" "$ACTOR_SEED" "$PARENT_REWARD_ALIGNMENT"
import json, sys
screen = json.load(open(sys.argv[1]))
health = json.loads(sys.argv[2])
capabilities = json.loads(sys.argv[3])
preregistration = json.load(open(sys.argv[4]))
actor_seed = int(sys.argv[5])
assert screen["passed"] is True, screen
assert health["status"] == "ready", health
assert health["model_version"] == capabilities["model_version"], (health, capabilities)
registered_parent = preregistration.get("parent", {}).get("model_version")
if registered_parent is not None:
    assert registered_parent == health["model_version"], (registered_parent, health)
registered_seed = preregistration.get("official_training", {}).get("actor_seed")
if registered_seed is not None:
    assert int(registered_seed) == actor_seed, (registered_seed, actor_seed)
if sys.argv[6]:
    reward_alignment = json.load(open(sys.argv[6]))
    assert reward_alignment["format"] == "strict-track2-frozen-official-reward-alignment-audit-v1"
PY
PYTHONPATH="$RUNTIME_PYTHONPATH" "$PYTHON_BIN" -c \
  'import diffsynth, openpi, rlinf' >/dev/null
mkdir -p "$RUN_ROOT/audit"
cp "$PREREGISTRATION" "$RUN_ROOT/audit/"
cp "$JOINT_GOAL" "$RUN_ROOT/audit/"
cp "$PARENT_PREREGISTRATION" "$RUN_ROOT/audit/"
cp "$PARENT_SCREEN" "$RUN_ROOT/audit/selected_parent_screen.json"
if [[ -n "$PARENT_REWARD_ALIGNMENT" ]]; then
  cp "$PARENT_REWARD_ALIGNMENT" "$RUN_ROOT/audit/selected_parent_reward_alignment.json"
fi
{
  printf 'actor_seed=%s\n' "$ACTOR_SEED"
  printf 'run_label=%s\n' "$RUN_LABEL"
  printf 'config_sha256=%s\n' "${EXPECTED_HASHES[$CONFIG]}"
  printf 'env_config_sha256=%s\n' "${EXPECTED_HASHES[$ENV_CONFIG]}"
  printf 'openpi_sha256=%s\n' "${EXPECTED_HASHES[$OPENPI_CKPT/model.safetensors]}"
  printf 'reward_sha256=%s\n' "${EXPECTED_HASHES[$REWARD_CKPT]}"
  printf 't5_model_sha256=%s\n' "${EXPECTED_HASHES[$T5_CKPT/model.safetensors]}"
  printf 'global_steps=1\n'
  printf 'transport_device=cpu\n'
  printf 'mpc=false\n'
  printf 'parent_screen=%s\n' "$PARENT_SCREEN"
  printf 'parent_screen_sha256=%s\n' "$(sha256sum "$PARENT_SCREEN" | cut -d' ' -f1)"
  if [[ -n "$PARENT_REWARD_ALIGNMENT" ]]; then
    printf 'parent_reward_alignment=%s\n' "$PARENT_REWARD_ALIGNMENT"
    printf 'parent_reward_alignment_sha256=%s\n' "$(sha256sum "$PARENT_REWARD_ALIGNMENT" | cut -d' ' -f1)"
  fi
  printf 'policy_preregistration=%s\n' "$PREREGISTRATION"
  printf 'policy_preregistration_sha256=%s\n' "$(sha256sum "$PREREGISTRATION" | cut -d' ' -f1)"
  printf 'world_model_health=%s\n' "$WORLD_MODEL_HEALTH"
  printf 'world_model_capabilities=%s\n' "$WORLD_MODEL_CAPABILITIES"
} >"$RUN_ROOT/audit/run_contract.txt"

"$PYTHON_BIN" -m ray.scripts.scripts stop --force \
  >"$RUN_ROOT/ray_stop_before.log" 2>&1 || true
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
  --config-name wan_robotwin_adjust_bottle_http_grpo_openpi_pi05 \
  runner.logger.log_path="$RUN_ROOT" \
  actor.seed="$ACTOR_SEED" \
  +weight_syncer.patch.transport_device=cpu >"$RUN_ROOT/launcher.log" 2>&1

grep -aq 'Global Step:    1/1' "$RUN_ROOT/launcher.log"
test -s "$CHECKPOINT"
find "$RUN_ROOT/$EXPERIMENT/checkpoints/global_step_1/actor" -type f -print0 \
  | sort -z | xargs -0 sha256sum >"$RUN_ROOT/audit/checkpoint_sha256.txt"
printf 'COMPLETE actor_seed=%s checkpoint=%s\n' "$ACTOR_SEED" "$CHECKPOINT"
