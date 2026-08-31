#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge';OFF="$BASE/artifacts/strict_track2_official_20260810";P="$BASE/pipeline/scripts";PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819';RUN="$OFF/runs/$NAME";REG="$OFF/run_registry/$NAME";DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403";OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403";VARIANT='v275_v274r1_fullbudget_step5_seed1471';CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt";TRAIN="$RUN/audit/p3_training_acceptance.json";EVAL_LOG="$OUT/$VARIANT/batch_00/launcher.log";PREREG="$REG/public_batch00_preregistration.json";REPORT="$RUN/audit/public_batch00_acceptance.json";STATIC="$RUN/audit/v275_public112_static_preflight.json"
test -f "$RUN/audit/V274_RETRY1_TRAINING_ACCEPTED";test -s "$STATIC";test ! -e "$OUT/$VARIANT";test ! -e "$PREREG"
"$PY" - "$STATIC" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['passed'] is True and x['outcomes_read'] is False and x['official_submission'] is False
PY
"$PY" "$P/prepare_v275_v274_batch00_gate.py" "$PREREG" "$CKPT" "$TRAIN" "$DEV/manifest.json" "$DEV/batch_00.json" "$VARIANT" >"$REG/public_batch00_prepare.log"
restore(){ "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after_public_batch00.log" 2>&1||true;bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_after_public_batch00.log" 2>&1||true;};trap restore EXIT
bash "$P/restart_v271_v274_services.sh" stop;bash "$P/restart_v218_services.sh" stop;"$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_before_public_batch00.log" 2>&1||true
TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$OUT" TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_batch00_eval.log" 2>&1
set +e;"$PY" "$P/audit_v228_public_right_gate.py" "$PREREG" "$EVAL_LOG" "$REPORT" >"$RUN/audit/public_batch00_acceptance.log" 2>&1;rc=$?;set -e
if ((rc));then touch "$RUN/audit/PUBLIC_BATCH00_REJECTED";exit "$rc";fi;touch "$RUN/audit/PUBLIC_BATCH00_PASSED";echo V275_V274R1_PUBLIC_BATCH00_PASSED
