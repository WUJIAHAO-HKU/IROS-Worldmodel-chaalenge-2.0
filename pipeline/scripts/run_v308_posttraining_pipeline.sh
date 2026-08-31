#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
NAME='v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
DEV_OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v309_v308_v301_rtx5090_step10_seed1471'
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
PUBLIC_PREREG="$REG/public_batch00_preregistration.json"
BATCH00_REPORT="$RUN/audit/public_batch00_acceptance.json"
PUBLIC112_REPORT="$RUN/audit/public112_acceptance.json"
POST_PREREG="$REG/posttraining_pipeline_preregistration.json"
PARENT_MANIFEST="$JOINT/v303_v301_batched_gates_seed1482_20260821/release_registration.json"
PARENT_RUNTIME="$BASE/pipeline/wam_pipeline/v301_batched_terminal_frame_mirror_runtime.py"
PUBLIC_PREFLIGHT="$OFF/runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v304_public112_static_preflight.json"
FINAL_PREFLIGHT="$OFF/runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v304_final128_static_preflight.json"
NUMERIC="$OFF/runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v303_numeric_gate_clarification.json"
FREEZE_DIR="$OFF/frozen_candidates/v311_v308_v301_rtx5090_step10_seed1471"
FREEZE="$FREEZE_DIR/freeze_manifest.json"
FINAL_OUTPUT="$OFF/real_robotwin_eval/frozen_v311_v308_v301_rtx5090_step10_seed1471_final128"
FINAL_PREREG="$FREEZE_DIR/final128_preregistration.json"
SEEDS="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_WRAPPER="$P/run_frozen_v311_v308_final128_once.sh"
CPUSET='0-21'

test -f "$RUN/audit/V308_TRAINING_ACCEPTED"
for path in "$CKPT" "$RUN/audit/p3_training_acceptance.json" "$REG/preregistration.json" "$POST_PREREG" "$PARENT_MANIFEST" "$PUBLIC_PREFLIGHT" "$FINAL_PREFLIGHT" "$NUMERIC" "$DEV/manifest.json" "$DEV/batch_00.json" "$SEEDS"; do test -s "$path"; done
test ! -e "$DEV_OUT/$VARIANT"
test ! -e "$FREEZE_DIR"
test ! -e "$FINAL_OUTPUT"

"$PY" "$P/verify_v304_posttraining_tools.py" --preregistration "$POST_PREREG" \
  --tool run_v308_posttraining_pipeline.sh \
  --tool run_frozen_v311_v308_final128_once.sh \
  --tool prepare_v305_v304_batch00_gate.py \
  --tool audit_v228_public_right_gate.py \
  --tool audit_v276_v274_public112_gate.py \
  --tool freeze_v307_v304_candidate.py \
  --tool prepare_frozen_final128_prereg.py \
  --tool run_strict_track2_dev_eval.sh \
  --tool run_strict_track2_final128_eval.sh \
  --tool summarize_frozen_final128.py >"$REG/posttraining_tool_verification.log"

"$PY" "$P/prepare_v305_v304_batch00_gate.py" \
  --output "$PUBLIC_PREREG" --checkpoint "$CKPT" \
  --training-report "$RUN/audit/p3_training_acceptance.json" \
  --training-preregistration "$REG/preregistration.json" \
  --world-release "$PARENT_MANIFEST" --static-preflight "$PUBLIC_PREFLIGHT" \
  --manifest "$DEV/manifest.json" --batch00 "$DEV/batch_00.json" \
  --prepare-script "$P/prepare_v305_v304_batch00_gate.py" \
  --launcher "$P/run_v308_posttraining_pipeline.sh" --variant "$VARIANT" >"$REG/public_batch00_prepare.log"

restore() {
  "$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_after_posttraining.log" 2>&1 || true
  TRACK2_CPUSET=0-7 bash "$P/restart_v271_v274_services.sh" start >"$REG/restart_v271_after_posttraining.log" 2>&1 || true
}
trap restore EXIT
bash "$P/restart_v308_rtx5090_services.sh" stop || true
bash "$P/restart_v271_v274_services.sh" stop || true
"$PY" -m ray.scripts.scripts stop --force >"$REG/ray_stop_before_public_batch00.log" 2>&1 || true

TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$DEV_OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 TRACK2_SKIP_BASELINE=true \
  taskset -c "$CPUSET" bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_batch00_eval.log" 2>&1

set +e
"$PY" "$P/audit_v228_public_right_gate.py" "$PUBLIC_PREREG" "$DEV_OUT/$VARIANT/batch_00/launcher.log" "$BATCH00_REPORT" >"$RUN/audit/public_batch00_acceptance.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then touch "$RUN/audit/PUBLIC_BATCH00_REJECTED"; exit "$rc"; fi
touch "$RUN/audit/PUBLIC_BATCH00_PASSED"

for batch in 01 02 03 04 05 06; do
  TRACK2_DEV_SEED_ROOT="$DEV" TRACK2_DEV_OUTPUT_ROOT="$DEV_OUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CKPT" TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" TRACK2_SKIP_BASELINE=true \
    taskset -c "$CPUSET" bash "$P/run_strict_track2_dev_eval.sh" >"$REG/public_batch_${batch}_eval.log" 2>&1
done

set +e
"$PY" "$P/audit_v276_v274_public112_gate.py" --preregistration "$PUBLIC_PREREG" --batch00-report "$BATCH00_REPORT" --output-root "$DEV_OUT" --dev-root "$DEV" --variant "$VARIANT" --output "$PUBLIC112_REPORT" >"$RUN/audit/public112_acceptance.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then touch "$RUN/audit/PUBLIC112_REJECTED"; exit "$rc"; fi
touch "$RUN/audit/PUBLIC112_PASSED"

"$PY" "$P/freeze_v307_v304_candidate.py" \
  --checkpoint "$CKPT" --training-report "$RUN/audit/p3_training_acceptance.json" \
  --training-preregistration "$REG/preregistration.json" --posttraining-preregistration "$POST_PREREG" \
  --public-preregistration "$PUBLIC_PREREG" --batch00-report "$BATCH00_REPORT" --public112-report "$PUBLIC112_REPORT" \
  --parent-manifest "$PARENT_MANIFEST" --parent-runtime "$PARENT_RUNTIME" --numeric-gate-clarification "$NUMERIC" \
  --public-runner "$P/run_strict_track2_dev_eval.sh" --final-runner "$P/run_strict_track2_final128_eval.sh" \
  --final-summarizer "$P/summarize_frozen_final128.py" --final-preregistration-tool "$P/prepare_frozen_final128_prereg.py" \
  --final-wrapper "$FINAL_WRAPPER" --public112-launcher "$P/run_v308_posttraining_pipeline.sh" \
  --final-static-preflight "$FINAL_PREFLIGHT" --public-static-preflight "$PUBLIC_PREFLIGHT" \
  --variant "$VARIANT" --output "$FREEZE" >"$RUN/audit/v311_freeze.log" 2>&1
touch "$RUN/audit/PUBLIC112_PASSED_AND_FROZEN"

"$PY" "$P/prepare_frozen_final128_prereg.py" --freeze-manifest "$FREEZE" --final-seed-manifest "$SEEDS" --output-root "$FINAL_OUTPUT" --output "$FINAL_PREREG" >"$FREEZE_DIR/final128_preregistration.log" 2>&1
exec bash "$FINAL_WRAPPER"
