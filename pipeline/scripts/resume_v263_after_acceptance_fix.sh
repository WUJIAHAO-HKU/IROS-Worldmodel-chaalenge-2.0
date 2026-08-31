#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v263_v261_fullright_chunk8_sft2048_joint4_grip4_lr1e6_seed1463_20260819'
VARIANT='v263_fullright_chunk8_sft2048_j4g4_seed1463'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
EVAL_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"

test -s "$CKPT"
test -f "$REG/preregistration.json"
test ! -e "$EVAL_OUT/$VARIANT"
test -f "$RUN/audit/extra_sft_acceptance.json"
cp "$RUN/audit/extra_sft_acceptance.json" \
  "$RUN/audit/extra_sft_acceptance_pre_dynamic_lr_fix.json"

"$PY" "$P/verify_v258_extra_sft.py" "$RUN" "$REG/preregistration.json" \
  "$CKPT" "$RUN/audit/extra_sft_acceptance.json" \
  >"$RUN/audit/extra_sft_acceptance.log" 2>&1
touch "$RUN/audit/V263_EXTRA_SFT_ACCEPTED_AFTER_DYNAMIC_LR_FIX"

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/resume_ray_stop_after.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/resume_restart_v218_after.log" 2>&1 || true
}
trap restore_services EXIT

bash "$P/restart_v218_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force >"$RUN/audit/resume_ray_stop_before_public_r0.log" 2>&1 || true

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
echo V263_PUBLIC_RIGHT_R1_PASSED
