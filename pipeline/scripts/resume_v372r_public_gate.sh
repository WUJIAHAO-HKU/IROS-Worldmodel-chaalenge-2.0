#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
P="$ROOT/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v372r_v371_deterministic_checkpoint_recovery_seed1535_20260823'
VARIANT='v372r_armconsistent_transitionbalanced_epoch1_seed1535'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
EVAL_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
SCREEN_NAME='v372r_public_gate'

restore_services() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after_public_gate.log" 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restart_v218_after_public_gate.log" 2>&1 || true
}

if [[ "${1:-}" != '--run' ]]; then
  test -s "$CKPT"
  test -f "$RUN/audit/V259_EXTRA_SFT_ACCEPTED"
  "$PY" - <<'PY' "$RUN/audit/extra_sft_acceptance.json"
import json, sys
assert json.load(open(sys.argv[1]))["accepted"] is True
PY
  screen -dmS "$SCREEN_NAME" taskset -c 0-21 "$0" --run
  sleep 3
  screen -ls | grep -q "$SCREEN_NAME"
  echo V372R_PUBLIC_GATE_STARTED
  exit 0
fi

trap restore_services EXIT
for screen_name in wm_v254_bridge wm_v254_gpu; do
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true
done
"$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_before_public_r0.log" 2>&1 || true

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
echo V372R_PUBLIC_RIGHT_R1_PASSED
