#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v219_v218_conservativekl_h200_r2_step8_lr5e6_beta005_seed1418_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RELEASE="$JOINT/v218_v217_route_aware_service_promotion_seed1417"
MANIFEST="$RELEASE/release_registration.json"
MODEL_VERSION='track2-v218-public-knn-blend-alpha070-route-aware'
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUTPUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v219_step8_seed1418'

restart_v218() {
  bash "$P/restart_v218_services.sh" start > "$REG/restart_v218_after_pipeline.log" 2>&1 || true
}
trap restart_v218 EXIT

"$PY" "$P/prepare_v219_rl.py" > "$REG.prepare.tmp.log" 2>&1
mv "$REG.prepare.tmp.log" "$REG/prepare.log"

curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q "$MODEL_VERSION"

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION="$MODEL_VERSION"
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=8 TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=8
export TRACK2_ACTOR_SEED=1418 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=5e-6 TRACK2_KL_BETA=0.05 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200
export TRACK2_ROLLOUT_EPOCH=2 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=400
export TRACK2_REFERENCE_STATE_STORAGE=disk
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TOKENIZERS_PARALLELISM=false

bash "$P/run_strict_track2_conservative_kl.sh" > "$REG/launcher.screen.log" 2>&1

"$PY" "$P/audit_strict_track2_kl_smoke.py" \
  --run "$RUN" --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/p3_training_acceptance.json" \
  > "$RUN/audit/p3_training_acceptance.log" 2>&1

"$PY" "$P/prepare_p3_public_policy_screen.py" \
  --checkpoint "$CHECKPOINT" \
  --training-audit "$RUN/audit/p3_training_acceptance.json" \
  --training-preregistration "$REG/preregistration.json" \
  --dev32-manifest "$DEV32/manifest.json" \
  --dev112-manifest "$DEV112/manifest.json" \
  --variant "$VARIANT" --output "$REG/public_policy_screen_preregistration.json" \
  > "$REG/public_policy_screen_preregistration.log"

for session in wm_v218_bridge wm_v218_gpu; do
  screen -S "$session" -X quit >/dev/null 2>&1 || true
done
"$PY" -m ray.scripts.scripts stop --force > "$RUN/audit/ray_stop_before_public_stage_a.log" 2>&1 || true

for batch in 00 01; do
  TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" bash "$P/run_strict_track2_dev_eval.sh"
done

"$PY" "$P/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" --dev-root "$DEV32" --candidate "$VARIANT" \
  --output "$RUN/audit/public_policy_stage_a_summary.json" \
  > "$RUN/audit/public_policy_stage_a_summary.log"
set +e
"$PY" "$P/audit_p3_public_policy_screen.py" \
  --preregistration "$REG/public_policy_screen_preregistration.json" \
  --summary "$RUN/audit/public_policy_stage_a_summary.json" --stage stage_a \
  --output "$RUN/audit/public_policy_stage_a_acceptance.json" \
  > "$RUN/audit/public_policy_stage_a_acceptance.log" 2>&1
stage_a_rc=$?
set -e
if (( stage_a_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"
  exit "$stage_a_rc"
fi
touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"

for batch in 02 03 04 05 06; do
  TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" bash "$P/run_strict_track2_dev_eval.sh"
done

"$PY" "$P/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" --dev-root "$DEV112" --candidate "$VARIANT" \
  --output "$RUN/audit/public_policy_stage_b_summary.json" \
  > "$RUN/audit/public_policy_stage_b_summary.log"
set +e
"$PY" "$P/audit_p3_public_policy_screen.py" \
  --preregistration "$REG/public_policy_screen_preregistration.json" \
  --summary "$RUN/audit/public_policy_stage_b_summary.json" --stage stage_b \
  --output "$RUN/audit/public_policy_stage_b_acceptance.json" \
  > "$RUN/audit/public_policy_stage_b_acceptance.log" 2>&1
stage_b_rc=$?
set -e
if (( stage_b_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
  exit "$stage_b_rc"
fi
touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
echo V219_PUBLIC_POLICY_STAGE_B_PASSED
