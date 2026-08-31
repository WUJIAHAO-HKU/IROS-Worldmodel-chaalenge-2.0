#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLROOT="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v285b_v278_motionweighted_rightterminal_epoch1_b4_lr1e6_seed1478_retry1_20260821'
VARIANT='v285b_v278_motionweighted_rightterminal_seed1478_retry1'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
BASE_CKPT="$OFF/runs/v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
DATA='/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_right_terminal_v1'
DATA_AUDIT="$OFF/run_registry/v225_right_terminal_dataset_20260818/dataset_audit.json"
CONVERSION="$DATA/conversion_summary.json"
TEMPORAL_AUDIT="$OFF/diagnostics/right_terminal_temporal_loss_audit_20260819.json"
ACTOR_SOURCE="$RLROOT/rlinf/workers/actor/fsdp_actor_worker.py"
RUNNER="$P/run_strict_track2_conservative_kl.sh"
MOTION_TEST="$P/test_motion_weighted_sft_contract.py"
RELEASE="$JOINT/v255_v254_delta_regime_service_seed1457_20260819"
MANIFEST="$RELEASE/release_registration.json"
MODEL_VERSION='track2-v254-delta-regime-terminal'
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
EVAL_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"

test ! -e "$RUN"
test ! -e "$REG"
test ! -e "$EVAL_OUT/$VARIANT"
test -s "$BASE_CKPT"
test -f "$DATA_AUDIT"
test -f "$CONVERSION"
test -f "$TEMPORAL_AUDIT"

"$PY" "$P/test_motion_weighted_sft_contract.py" "$ACTOR_SOURCE"
"$PY" "$P/prepare_v285_motion_weighted_right_sft.py" \
  "$REG/preregistration.json" "$BASE_CKPT" "$DATA_AUDIT" "$CONVERSION" \
  "$TEMPORAL_AUDIT" "$ACTOR_SOURCE" "$RUNNER" "$MOTION_TEST" "$NAME" \
  >"$OFF/run_registry/$NAME.prepare.tmp.log" 2>&1
mv "$OFF/run_registry/$NAME.prepare.tmp.log" "$REG/prepare.log"

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after.log" 2>&1 || true
  bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_v274_after.log" 2>&1 || true
}
trap restore_services EXIT

for screen_name in wm_v271_v274_bridge wm_v271_v274_gpu; do
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 30); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:8005/v1/health >/dev/null 2>&1 \
    && ! curl -fsS --max-time 1 http://127.0.0.1:18084/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/restart_v254_services.sh" start \
  >"$REG/restart_v254_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST" TRACK2_CKPT_PATH="$BASE_CKPT"
export TRACK2_MAX_STEPS=1 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1478 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=1e-6 TRACK2_KL_BETA=0.1 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=8 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=8
export TRACK2_ROLLOUT_EPOCH=1 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=8
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TRACK2_ENABLE_SFT_CO_TRAIN=true TRACK2_SFT_DATA_PATH="$DATA"
export TRACK2_SFT_CONFIG_NAME='pi05_aloha_robotwin_head_adjust_bottle'
export TRACK2_SFT_LOSS_WEIGHT=0.3 TRACK2_SFT_BATCH_SIZE=4 TRACK2_SFT_NUM_WORKERS=0
export TRACK2_SFT_SINGLE_ACTIVE_ARM=1 TRACK2_USE_STRUCTURED_SFT_ACTION_LOSS=true
export TRACK2_USE_MIXED_ARM_STRUCTURED_SFT_ACTION_LOSS=false
export TRACK2_SFT_USE_ACTION_CHUNK_LOSS=true
export TRACK2_SFT_ACTIVE_JOINT_WEIGHT=4.0 TRACK2_SFT_ACTIVE_GRIPPER_WEIGHT=1.0
export TRACK2_SFT_INACTIVE_KEEP_WEIGHT=2.0
export TRACK2_SFT_USE_MOTION_SAMPLE_WEIGHTING=true
export TRACK2_SFT_MOTION_WEIGHT_HORIZON=8 TRACK2_SFT_MOTION_WEIGHT_REFERENCE=0.05
export TRACK2_SFT_MOTION_MINIMUM_WEIGHT=0.05 TRACK2_SFT_MOTION_WEIGHT_POWER=1.0
export TRACK2_SFT_EXTRA_UPDATES_PER_GLOBAL_BATCH=315 TRACK2_SFT_EXTRA_LOSS_WEIGHT=1.0
export TRACK2_CPUSET=0-23 TOKENIZERS_PARALLELISM=false

bash "$RUNNER" >"$REG/launcher.screen.log" 2>&1
"$PY" "$P/verify_v285_motion_sft.py" "$RUN" "$REG/preregistration.json" \
  "$CKPT" "$RUN/audit/motion_sft_acceptance.json" \
  >"$RUN/audit/motion_sft_acceptance.log" 2>&1
touch "$RUN/audit/V285_MOTION_SFT_ACCEPTED"

for screen_name in wm_v254_bridge wm_v254_gpu; do
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true
done
"$PY" -m ray.scripts.scripts stop --force >"$RUN/audit/ray_stop_before_public_batch00.log" 2>&1 || true

TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$EVAL_OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_batch00_eval.log" 2>&1

set +e
"$PY" "$P/audit_v228_public_right_gate.py" "$REG/preregistration.json" \
  "$EVAL_OUT/$VARIANT/batch_00/launcher.log" \
  "$RUN/audit/public_batch00_acceptance.json" \
  >"$RUN/audit/public_batch00_acceptance.log" 2>&1
gate_rc=$?
set -e
if (( gate_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_BATCH00_REJECTED"
  exit "$gate_rc"
fi
touch "$RUN/audit/PUBLIC_BATCH00_PASSED"
echo V285_PUBLIC_BATCH00_PASSED
