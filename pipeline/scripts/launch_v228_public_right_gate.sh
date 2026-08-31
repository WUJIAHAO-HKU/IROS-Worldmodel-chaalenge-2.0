#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME="${TRACK2_GATE_NAME:-v228_v223_terminal_rightsft_cachefix_h8_step64_seed1426_20260818}"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT="${TRACK2_GATE_VARIANT:-v228_v223_terminal_rightsft_cachefix_step64_seed1426}"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_64/actor/model_state_dict/full_weights.pt"
TRAIN_REPORT="$RUN/audit/terminal_right_sft_retry_acceptance.json"
EVAL_LOG="$OUT/$VARIANT/batch_00/launcher.log"
PREREG="$REG/public_right_gate_preregistration.json"
REPORT="$RUN/audit/public_right_gate_acceptance.json"

TRAIN_ACCEPTED_MARKER="${TRACK2_GATE_TRAIN_MARKER:-V228_TERMINAL_RIGHT_SFT_RETRY_ACCEPTED}"
test -f "$RUN/audit/$TRAIN_ACCEPTED_MARKER"
test ! -e "$OUT/$VARIANT"
test ! -e "$PREREG"
"$PY" "$P/prepare_v228_public_right_gate.py" "$PREREG" "$CKPT" "$TRAIN_REPORT" \
  "$DEV/manifest.json" "$DEV/batch_00.json" "$VARIANT" > "$REG/public_right_gate_prepare.log"

restore() {
  "$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_after_public_right_gate.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start > "$REG/restart_v218_after_public_right_gate.log" 2>&1 || true
}
trap restore EXIT
bash "$P/restart_v218_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_before_public_right_gate.log" 2>&1 || true
TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$OUT" TRACK2_CANDIDATE_CHECKPOINT="$CKPT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" > "$REG/public_right_gate_eval.log" 2>&1
set +e
"$PY" "$P/audit_v228_public_right_gate.py" "$PREREG" "$EVAL_LOG" "$REPORT" \
  > "$RUN/audit/public_right_gate_acceptance.log" 2>&1
rc=$?
set -e
if ((rc)); then
  touch "$RUN/audit/PUBLIC_RIGHT_GATE_REJECTED"
  exit "$rc"
fi
touch "$RUN/audit/PUBLIC_RIGHT_GATE_PASSED"
echo V228_PUBLIC_RIGHT_GATE_PASSED
