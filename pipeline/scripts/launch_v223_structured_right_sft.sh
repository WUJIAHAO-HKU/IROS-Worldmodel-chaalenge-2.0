#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
P="$BASE/pipeline/scripts"
RLROOT="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v223_v169_structured_rightsft_kl_h8_step24_seed1423_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
MODEL_VERSION='track2-v218-public-knn-blend-alpha070-route-aware'
PARENT="$JOINT/v218_v217_route_aware_service_promotion_seed1417/release_registration.json"
BASE_CKPT="$OFF/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
SFT_DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1'
SMOKE_REPORT="$OFF/runs/v222_v169_structured_rightsft_smoke_h8_step1_seed1422_20260818/audit/structured_sft_smoke_acceptance.json"
OUTPUT_CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_24/actor/model_state_dict/full_weights.pt"

test ! -e "$RUN"
test ! -e "$REG"
test -f "$OFF/runs/v222_v169_structured_rightsft_smoke_h8_step1_seed1422_20260818/audit/V222_STRUCTURED_SFT_SMOKE_ACCEPTED"
df --output=avail -B1 /root/autodl-tmp | tail -1 | awk '$1 >= 20000000000 {ok=1} END {exit !ok}'
"$PY" "$P/prepare_v223_structured_right_sft.py" "$REG/preregistration.json" \
  "$BASE_CKPT" "$SFT_DATA" "$RLROOT/rlinf/workers/actor/fsdp_actor_worker.py" \
  "$RLROOT/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$P/run_strict_track2_conservative_kl.sh" "$SMOKE_REPORT" \
  > "/tmp/${NAME}.prepare.log"
mv "/tmp/${NAME}.prepare.log" "$REG/prepare.log"

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_after.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start > "$REG/restart_v218_after.log" 2>&1 || true
}
trap restore_services EXIT
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'

export TRACK2_ROOT="$BASE" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$PARENT" TRACK2_CKPT_PATH="$BASE_CKPT"
export TRACK2_MAX_STEPS=24 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1423 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=1e-5 TRACK2_KL_BETA=0.1 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=8 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=8
export TRACK2_ROLLOUT_EPOCH=1 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=8
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TRACK2_ENABLE_SFT_CO_TRAIN=true TRACK2_SFT_DATA_PATH="$SFT_DATA"
export TRACK2_SFT_CONFIG_NAME='pi05_aloha_robotwin_head_adjust_bottle'
export TRACK2_SFT_LOSS_WEIGHT=0.1 TRACK2_SFT_BATCH_SIZE=2 TRACK2_SFT_NUM_WORKERS=0
export TRACK2_SFT_SINGLE_ACTIVE_ARM=1 TRACK2_USE_STRUCTURED_SFT_ACTION_LOSS=true
export TRACK2_SFT_ACTIVE_JOINT_WEIGHT=1.0 TRACK2_SFT_ACTIVE_GRIPPER_WEIGHT=3.0
export TRACK2_SFT_INACTIVE_KEEP_WEIGHT=0.25
export TOKENIZERS_PARALLELISM=false

bash "$P/run_strict_track2_conservative_kl.sh" > "$REG/launcher.screen.log" 2>&1
"$PY" "$P/verify_v223_structured_right_sft.py" "$RUN" "$OUTPUT_CKPT" 24 \
  > "$RUN/audit/structured_right_sft_training_acceptance.log"
grep -aE 'sft_loss|weighted_sft_loss|sft_arm|sft_physical_action_weight_sum|actor/kl_loss|actor/total_loss|actor/grad_norm' \
  "$RUN/launcher.log" > "$RUN/audit/structured_sft_training_metrics_all.txt"
echo V223_STRUCTURED_RIGHT_SFT_TRAINING_ACCEPTED
