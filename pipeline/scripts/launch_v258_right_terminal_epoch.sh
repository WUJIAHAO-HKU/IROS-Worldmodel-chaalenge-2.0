#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLROOT="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME="${TRACK2_V258_NAME:-v258_v169step5_rightterminal_sft1epoch_h8_step1_extra638_lr5e6_seed1460_20260819}"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
BASE_CKPT="$OFF/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_right_terminal_v1'
DATA_AUDIT="$OFF/run_registry/v225_right_terminal_dataset_20260818/dataset_audit.json"
CONVERSION="$DATA/conversion_summary.json"
ACTOR_SOURCE="$RLROOT/rlinf/workers/actor/fsdp_actor_worker.py"
RUNNER="$P/run_strict_track2_conservative_kl.sh"
RELEASE="$JOINT/v255_v254_delta_regime_service_seed1457_20260819"
MANIFEST="$RELEASE/release_registration.json"
MODEL_VERSION='track2-v254-delta-regime-terminal'
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
EVAL_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT="${TRACK2_V258_VARIANT:-v258_rightterminal_epoch1_seed1460}"
RETRY_OF="${TRACK2_V258_RETRY_OF:-}"

test ! -e "$RUN"
test ! -e "$REG"
test ! -e "$EVAL_OUT/$VARIANT"
test -s "$BASE_CKPT"
test -f "$DATA_AUDIT"
test -f "$CONVERSION"

prepare_args=(
  "$REG/preregistration.json" "$BASE_CKPT" "$DATA_AUDIT" "$CONVERSION"
  "$ACTOR_SOURCE" "$RUNNER" "$NAME"
)
if [[ -n "$RETRY_OF" ]]; then
  prepare_args+=("$RETRY_OF")
fi
"$PY" "$P/prepare_v258_right_terminal_epoch.py" "${prepare_args[@]}" \
  >"$OFF/run_registry/$NAME.prepare.tmp.log" 2>&1
mv "$OFF/run_registry/$NAME.prepare.tmp.log" "$REG/prepare.log"

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restart_v218_after.log" 2>&1 || true
}
trap restore_services EXIT

TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/restart_v254_services.sh" start \
  >"$REG/restart_v254_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST" TRACK2_CKPT_PATH="$BASE_CKPT"
export TRACK2_MAX_STEPS=1 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1460 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=5e-6 TRACK2_KL_BETA=0.1 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=8 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=8
export TRACK2_ROLLOUT_EPOCH=1 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=8
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TRACK2_ENABLE_SFT_CO_TRAIN=true TRACK2_SFT_DATA_PATH="$DATA"
export TRACK2_SFT_CONFIG_NAME='pi05_aloha_robotwin_head_adjust_bottle'
export TRACK2_SFT_LOSS_WEIGHT=0.3 TRACK2_SFT_BATCH_SIZE=2 TRACK2_SFT_NUM_WORKERS=0
export TRACK2_SFT_SINGLE_ACTIVE_ARM=1 TRACK2_USE_STRUCTURED_SFT_ACTION_LOSS=true
export TRACK2_SFT_ACTIVE_JOINT_WEIGHT=1.0 TRACK2_SFT_ACTIVE_GRIPPER_WEIGHT=3.0
export TRACK2_SFT_INACTIVE_KEEP_WEIGHT=0.25
export TRACK2_SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH=638 TRACK2_SFT_EXTRA_LOSS_WEIGHT=1.0
export TOKENIZERS_PARALLELISM=false

bash "$P/run_strict_track2_conservative_kl.sh" >"$REG/launcher.screen.log" 2>&1
"$PY" "$P/verify_v258_extra_sft.py" "$RUN" "$REG/preregistration.json" \
  "$CKPT" "$RUN/audit/extra_sft_acceptance.json" \
  >"$RUN/audit/extra_sft_acceptance.log" 2>&1
touch "$RUN/audit/V258_EXTRA_SFT_ACCEPTED"

for screen_name in wm_v254_bridge wm_v254_gpu; do
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true
done
"$PY" -m ray.scripts.scripts stop --force >"$RUN/audit/ray_stop_before_public_r0.log" 2>&1 || true

TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$EVAL_OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_r0_eval.log" 2>&1

set +e
"$PY" "$P/audit_v228_public_right_gate.py" "$REG/preregistration.json" \
  "$EVAL_OUT/$VARIANT/batch_00/launcher.log" \
  "$RUN/audit/public_right_r0_acceptance.json" \
  >"$RUN/audit/public_right_r0_acceptance.log" 2>&1
gate_rc=$?
set -e
if (( gate_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_RIGHT_R0_REJECTED"
  exit "$gate_rc"
fi
touch "$RUN/audit/PUBLIC_RIGHT_R0_PASSED"

TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$EVAL_OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=01 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_r1_eval.log" 2>&1

"$PY" "$P/summarize_strict_track2_dev_eval.py" --output-root "$EVAL_OUT" \
  --dev-root "$DEV32" --candidate "$VARIANT" \
  --output "$RUN/audit/public_right_r1_summary.json" \
  >"$RUN/audit/public_right_r1_summary.log" 2>&1
set +e
"$PY" "$P/audit_v258_stage_r1.py" "$REG/preregistration.json" \
  "$RUN/audit/public_right_r1_summary.json" \
  "$RUN/audit/public_right_r1_acceptance.json" \
  >"$RUN/audit/public_right_r1_acceptance.log" 2>&1
stage_rc=$?
set -e
if (( stage_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_RIGHT_R1_REJECTED"
  exit "$stage_rc"
fi
touch "$RUN/audit/PUBLIC_RIGHT_R1_PASSED"
echo V258_PUBLIC_RIGHT_R1_PASSED
