#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';OFF="$ROOT/artifacts/strict_track2_official_20260810";JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_20260819';RUN="$OFF/runs/$NAME";REG="$OFF/run_registry/$NAME";P="$ROOT/pipeline/scripts";PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';RELEASE="$JOINT/v273_v271_corrected_alpha_long_gate_seed1470_20260819";MANIFEST="$RELEASE/release_registration.json";MODEL_VERSION='track2-v271-endpoint-calibrated-terminal';CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt"
restore(){ "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1||true;bash "$P/restart_v271_v274_services.sh" stop >/dev/null 2>&1||true;bash "$P/restart_v218_services.sh" start >"$REG/restart_v218_after.log" 2>&1||true;};trap restore EXIT
"$PY" "$P/prepare_v274_v271_fullbudget_rl.py" >"$OFF/run_registry/$NAME.prepare.tmp.log" 2>&1;mv "$OFF/run_registry/$NAME.prepare.tmp.log" "$REG/prepare.log"
TRACK2_WAM_RELEASE_CUDA_CACHE=1 bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health|grep -q '"status":"ready"';curl -fsS http://127.0.0.1:8005/v1/health|grep -q "$MODEL_VERSION"
unset TRACK2_CKPT_PATH || true
export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN" TRACK2_MODEL_VERSION="$MODEL_VERSION" TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005' TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=5 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1471 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=2e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200 TRACK2_ROLLOUT_EPOCH=8 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=6400
export TRACK2_REFERENCE_STATE_STORAGE=disk TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0 TOKENIZERS_PARALLELISM=false
bash "$P/run_strict_track2_conservative_kl.sh" >"$REG/launcher.screen.log" 2>&1
"$PY" "$P/audit_strict_track2_kl_smoke.py" --run "$RUN" --preregistration "$REG/preregistration.json" --output "$RUN/audit/p3_training_acceptance.json" >"$RUN/audit/p3_training_acceptance.log" 2>&1
touch "$RUN/audit/V274_TRAINING_ACCEPTED"
test -s "$CHECKPOINT"
echo V274_FULLBUDGET_TRAINING_ACCEPTED
