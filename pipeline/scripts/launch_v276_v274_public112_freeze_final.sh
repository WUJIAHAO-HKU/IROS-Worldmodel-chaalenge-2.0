#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v274_v271_fresh_fullbudget_h200_r8_step5_lr2e5_beta001_seed1471_retry1_20260819'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v275_v274r1_fullbudget_step5_seed1471'
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt"
PREREG="$REG/public_batch00_preregistration.json"
BATCH00="$RUN/audit/public_batch00_acceptance.json"
PUBLIC112="$RUN/audit/public112_acceptance.json"
FREEZE_DIR="$OFF/frozen_candidates/v277_v274r1_fullbudget_step5_seed1471"
FREEZE="$FREEZE_DIR/freeze_manifest.json"
FINAL_OUTPUT="$OFF/real_robotwin_eval/frozen_v277_v274r1_fullbudget_step5_seed1471_final128"
FINAL_PREREG="$FREEZE_DIR/final128_preregistration.json"
SEED_BUNDLE="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_PREFLIGHT="$RUN/audit/v277_final128_static_preflight.json"
PUBLIC_PREFLIGHT="$RUN/audit/v275_public112_static_preflight.json"
FINAL_WRAPPER="$P/run_frozen_v277_v274r1_final128_once.sh"

test -f "$RUN/audit/PUBLIC_BATCH00_PASSED"
test ! -e "$RUN/audit/PUBLIC_BATCH00_REJECTED"
for path in "$CKPT" "$PREREG" "$BATCH00" "$DEV/manifest.json" "$SEED_BUNDLE" "$FINAL_PREFLIGHT" "$PUBLIC_PREFLIGHT";do test -s "$path";done
test ! -e "$RUN/audit/PUBLIC112_PASSED"
test ! -e "$RUN/audit/PUBLIC112_REJECTED"
test ! -e "$FREEZE"
if find "$OFF/frozen_candidates" -type f -name freeze_manifest.json -print -quit 2>/dev/null|grep -q .;then
  echo 'REFUSING_FREEZE: another final candidate already exists' >&2;exit 3
fi

restore(){ "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after_v276.log" 2>&1||true;bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_after_v276.log" 2>&1||true; }
trap restore EXIT
bash "$P/restart_v271_v274_services.sh" stop
"$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_before_public112_remainder.log" 2>&1||true

for batch in 01 02 03 04 05 06;do
  TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$OUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" TRACK2_SKIP_BASELINE=true \
    bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_batch_${batch}_eval.log" 2>&1
done

set +e
"$PY" "$P/audit_v276_v274_public112_gate.py" \
  --preregistration "$PREREG" --batch00-report "$BATCH00" \
  --output-root "$OUT" --dev-root "$DEV" --variant "$VARIANT" \
  --output "$PUBLIC112" >"$RUN/audit/public112_acceptance.log" 2>&1
public_rc=$?
set -e
if ((public_rc!=0));then touch "$RUN/audit/PUBLIC112_REJECTED";exit "$public_rc";fi
touch "$RUN/audit/PUBLIC112_PASSED"

"$PY" "$P/freeze_v277_v274_candidate.py" \
  --checkpoint "$CKPT" --training-report "$RUN/audit/p3_training_acceptance.json" \
  --public-preregistration "$PREREG" --batch00-report "$BATCH00" \
  --public112-report "$PUBLIC112" \
  --parent-manifest "$JOINT/v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json" \
  --parent-runtime "$P/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py" \
  --final-runner "$P/run_strict_track2_final128_eval.sh" \
  --final-summarizer "$P/summarize_frozen_final128.py" \
  --final-preregistration-tool "$P/prepare_frozen_final128_prereg.py" \
  --final-wrapper "$FINAL_WRAPPER" --final-static-preflight "$FINAL_PREFLIGHT" \
  --public-static-preflight "$PUBLIC_PREFLIGHT" \
  --variant "$VARIANT" --output "$FREEZE" \
  >"$RUN/audit/v277_freeze.log" 2>&1
touch "$RUN/audit/PUBLIC112_PASSED_AND_FROZEN"

"$PY" "$P/prepare_frozen_final128_prereg.py" \
  --freeze-manifest "$FREEZE" --final-seed-manifest "$SEED_BUNDLE" \
  --output-root "$FINAL_OUTPUT" --output "$FINAL_PREREG" \
  >"$FREEZE_DIR/final128_preregistration.log" 2>&1

bash "$FINAL_WRAPPER"
