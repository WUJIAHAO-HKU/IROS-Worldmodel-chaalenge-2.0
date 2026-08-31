#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v219_v218_conservativekl_h200_r2_step8_lr5e6_beta005_seed1418_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUTPUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v219_step8_seed1418'
PLANNER="$OFF/official_deps/RoboTwin_RLinf_support/envs/robot/planner.py"
PLANNER_SHA='178a72a7b6ededee66e7f78ebc35e7c39c4a855a1d713b90a01a32868d49f24a'

restart_v218() {
  bash "$P/restart_v218_services.sh" start >"$REG/restart_v218_after_public_resume.log" 2>&1 || true
}
trap restart_v218 EXIT

test -s "$CHECKPOINT"
test -s "$RUN/audit/p3_training_acceptance.json"
test -s "$REG/public_policy_screen_preregistration.json"
test -s "$REG/amendments/curobo_compat_pre_amendment.json"
test -s "$REG/amendments/curobo_compat_applied.json"
test "$(sha256sum "$PLANNER" | awk '{print $1}')" = "$PLANNER_SHA"
test ! -e "$RUN/audit/public_policy_stage_a_summary.json"
test ! -e "$RUN/audit/public_policy_stage_a_acceptance.json"

failed_log="$OUTPUT/$VARIANT/batch_00/launcher.log"
if [[ -e "$failed_log" ]] && grep -Fq 'KeyboardInterrupt' "$failed_log"; then
  mv "$failed_log" "$OUTPUT/$VARIANT/batch_00/launcher.pre_startup_observation_fix.log"
fi

for session in wm_v218_bridge wm_v218_gpu; do
  screen -S "$session" -X quit >/dev/null 2>&1 || true
done
"$PY" -m ray.scripts.scripts stop --force >"$RUN/audit/ray_stop_before_public_resume.log" 2>&1 || true

for batch in 00 01; do
  TRACK2_DEV_SEED_ROOT="$DEV112" TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" bash "$P/run_strict_track2_dev_eval.sh"
done

"$PY" "$P/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" --dev-root "$DEV32" --candidate "$VARIANT" \
  --output "$RUN/audit/public_policy_stage_a_summary.json" \
  >"$RUN/audit/public_policy_stage_a_summary.log"
set +e
"$PY" "$P/audit_p3_public_policy_screen.py" \
  --preregistration "$REG/public_policy_screen_preregistration.json" \
  --summary "$RUN/audit/public_policy_stage_a_summary.json" --stage stage_a \
  --output "$RUN/audit/public_policy_stage_a_acceptance.json" \
  >"$RUN/audit/public_policy_stage_a_acceptance.log" 2>&1
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
  >"$RUN/audit/public_policy_stage_b_summary.log"
set +e
"$PY" "$P/audit_p3_public_policy_screen.py" \
  --preregistration "$REG/public_policy_screen_preregistration.json" \
  --summary "$RUN/audit/public_policy_stage_b_summary.json" --stage stage_b \
  --output "$RUN/audit/public_policy_stage_b_acceptance.json" \
  >"$RUN/audit/public_policy_stage_b_acceptance.log" 2>&1
stage_b_rc=$?
set -e
if (( stage_b_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
  exit "$stage_b_rc"
fi
touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
echo V219_PUBLIC_POLICY_STAGE_B_PASSED
