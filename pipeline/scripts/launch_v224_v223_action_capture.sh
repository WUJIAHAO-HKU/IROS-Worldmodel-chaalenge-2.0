#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
V223="${TRACK2_CAPTURE_RUN:-v223_v169_structured_rightsft_kl_h8_step24_seed1423_20260818}"
RUN="$OFF/runs/$V223"
NAME="${TRACK2_CAPTURE_NAME:-v224_v223_public_batch00_action_capture_20260818}"
REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT="${TRACK2_CAPTURE_VARIANT:-v224_v223_same_batch00_action_capture}"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_${TRACK2_CAPTURE_STEP:-24}/actor/model_state_dict/full_weights.pt"
TRAIN_REPORT="$RUN/audit/${TRACK2_CAPTURE_TRAIN_REPORT:-structured_right_sft_training_acceptance.json}"
PRIOR_LOG="$OUT/${TRACK2_CAPTURE_PRIOR_VARIANT:-v223_v169_structured_rightsft_kl_step24_seed1423}/batch_00/launcher.log"
PRIOR_GATE="$RUN/audit/${TRACK2_CAPTURE_PRIOR_GATE_REPORT:-public_right_gate_acceptance.json}"
CAPTURE="$REG/action_capture"
CAPTURE_LOG="$OUT/$VARIANT/batch_00/launcher.log"

test ! -e "$REG"
test ! -e "$OUT/$VARIANT"
"$PY" "$P/prepare_v224_v223_action_capture.py" "$REG/preregistration.json" \
  "$CKPT" "$TRAIN_REPORT" "$DEV/batch_00.json" "$PRIOR_GATE" > /tmp/v224_prepare.log
mv /tmp/v224_prepare.log "$REG/prepare.log"
mkdir -p "$CAPTURE"

restore() {
  "$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_after.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start > "$REG/restart_v218_after.log" 2>&1 || true
}
trap restore EXIT
bash "$P/restart_v218_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force > "$REG/ray_stop_before.log" 2>&1 || true
TRACK2_ACTION_CAPTURE_DIR="$CAPTURE" \
TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
bash "$P/run_strict_track2_dev_eval.sh" > "$REG/action_capture_eval.log" 2>&1

"$PY" "$P/analyze_v224_v223_action_capture.py" "$CAPTURE" "$PRIOR_LOG" "$CAPTURE_LOG" \
  '/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/adjust_bottle_train40_arm1/conversion_summary.json' \
  "$REG/action_capture_analysis.json" > "$REG/action_capture_analysis.log"
touch "$REG/$(printf '%s' "$NAME" | cut -d_ -f1 | tr '[:lower:]' '[:upper:]')_ACTION_CAPTURE_COMPLETE"
echo ACTION_CAPTURE_COMPLETE
