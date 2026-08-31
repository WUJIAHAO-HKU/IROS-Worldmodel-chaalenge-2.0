#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
MANIFEST="$JOINT/v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json"

"$PY" "$P/prepare_v211_action_capture.py"
mkdir -p "$REG/bridge_audit"

restart_plain() {
  bash "$P/restart_v209_services.sh" start > "$REG/restart_v209_plain_after_diagnostic.log" 2>&1 || true
}
trap restart_plain EXIT

TRACK2_BRIDGE_AUDIT_DIR="$REG/bridge_audit" \
TRACK2_BRIDGE_AUDIT_MAX_ITEMS=50 \
bash "$P/restart_v209_services.sh" start > "$REG/restart_v209_capture.log" 2>&1

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION='track2-v209-public-arm-routed-v202-left-v208-right-selected'
export TRACK2_BRIDGE_URL='http://127.0.0.1:18083' TRACK2_WAM_URL='http://127.0.0.1:8004'
export TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=1 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1410 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=5e-6 TRACK2_KL_BETA=0.05 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200
export TRACK2_ROLLOUT_EPOCH=2 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=400
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TOKENIZERS_PARALLELISM=false

bash "$P/run_strict_track2_conservative_kl.sh" > "$REG/launcher.screen.log" 2>&1

"$PY" "$P/audit_v211_training_action_exploration.py" \
  --audit-dir "$REG/bridge_audit" \
  --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/training_action_exploration.json" \
  > "$RUN/audit/training_action_exploration.log"

touch "$RUN/audit/V211_ACTION_CAPTURE_COMPLETE"
echo V211_ACTION_CAPTURE_COMPLETE
