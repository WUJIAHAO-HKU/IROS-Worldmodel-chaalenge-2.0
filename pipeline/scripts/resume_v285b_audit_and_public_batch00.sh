#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v285b_v278_motionweighted_rightterminal_epoch1_b4_lr1e6_seed1478_retry1_20260821'
VARIANT='v285b_v278_motionweighted_rightterminal_seed1478_retry1'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
EVAL_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"

test -s "$CKPT"
test ! -e "$EVAL_OUT/$VARIANT"
test ! -e "$RUN/audit/V285_MOTION_SFT_ACCEPTED"

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after_public_batch00.log" 2>&1 || true
  bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_v274_after_public_batch00.log" 2>&1 || true
}
trap restore_services EXIT

"$PY" "$P/verify_v285_motion_sft.py" "$RUN" "$REG/preregistration.json" \
  "$CKPT" "$RUN/audit/motion_sft_acceptance.json" \
  >"$RUN/audit/motion_sft_acceptance.log" 2>&1
touch "$RUN/audit/V285_MOTION_SFT_ACCEPTED"

for screen_name in wm_v254_bridge wm_v254_gpu wm_v271_v274_bridge wm_v271_v274_gpu; do
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
echo V285B_PUBLIC_BATCH00_PASSED
