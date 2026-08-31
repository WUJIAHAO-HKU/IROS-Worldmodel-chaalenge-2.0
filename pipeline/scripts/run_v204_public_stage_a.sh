#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v204_v202_effectivekl_h200_r2_step8_lr1e5_beta005_seed1404_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
DEV32="$OFF/real_robotwin_eval/public_unseen_train_dev32_seed1403"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUTPUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v204_step8_seed1404'
SCREEN_PREREG="$REG/public_policy_screen_preregistration.json"
SUMMARY="$RUN/audit/public_policy_stage_a_summary.json"
AUDIT="$RUN/audit/public_policy_stage_a_acceptance.json"

test -s "$CHECKPOINT"
"$PY" - "$RUN/audit/p3_training_acceptance.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["passed"] is True
PY
test ! -e "$SCREEN_PREREG"
"$PY" "$BASE/pipeline/scripts/prepare_p3_public_policy_screen.py" \
  --checkpoint "$CHECKPOINT" \
  --training-audit "$RUN/audit/p3_training_acceptance.json" \
  --training-preregistration "$REG/preregistration.json" \
  --dev32-manifest "$DEV32/manifest.json" \
  --dev112-manifest "$DEV112/manifest.json" \
  --variant "$VARIANT" \
  --output "$SCREEN_PREREG" > "$REG/public_policy_screen_preregistration.log"

for session in wm_v202_bridge wm_v202_gpu; do
  screen -ls 2>/dev/null | grep -q "[.]$session" && screen -S "$session" -X quit || true
done
"$PY" -m ray.scripts.scripts stop --force > "$RUN/audit/ray_stop_before_public_stage_a.log" 2>&1 || true

for batch in 00 01; do
  TRACK2_DEV_SEED_ROOT="$DEV112" \
  TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
  TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" \
    bash "$BASE/pipeline/scripts/run_strict_track2_dev_eval.sh"
done

"$PY" "$BASE/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" --dev-root "$DEV32" --candidate "$VARIANT" \
  --output "$SUMMARY" > "$RUN/audit/public_policy_stage_a_summary.log"
set +e
"$PY" "$BASE/pipeline/scripts/audit_p3_public_policy_screen.py" \
  --preregistration "$SCREEN_PREREG" --summary "$SUMMARY" --stage stage_a \
  --output "$AUDIT" > "$RUN/audit/public_policy_stage_a_acceptance.log" 2>&1
rc=$?
set -e
if (( rc == 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"
else
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"
fi
exit "$rc"
