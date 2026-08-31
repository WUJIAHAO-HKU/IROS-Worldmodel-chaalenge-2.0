#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

BASE="/root/autodl-tmp/IROS_WAM_2.0 challenge"
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
P="$BASE/pipeline/scripts"
PY="/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
NAME="v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
DEV="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT="v305_v304_v301_fullbudget_step10_seed1471"
CKPT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
PREREG="$REG/public_batch00_preregistration.json"
BATCH00="$RUN/audit/public_batch00_acceptance.json"
PUBLIC112="$RUN/audit/public112_acceptance.json"
FREEZE_DIR="$OFF/frozen_candidates/v307_v304_v301_fullbudget_step10_seed1471"
FREEZE="$FREEZE_DIR/freeze_manifest.json"
FINAL_OUTPUT="$OFF/real_robotwin_eval/frozen_v307_v304_v301_fullbudget_step10_seed1471_final128"
FINAL_PREREG="$FREEZE_DIR/final128_preregistration.json"
SEED_BUNDLE="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_PREFLIGHT="$RUN/audit/v304_final128_static_preflight.json"
PUBLIC_PREFLIGHT="$RUN/audit/v304_public112_static_preflight.json"
NUMERIC_CLARIFICATION="$RUN/audit/v303_numeric_gate_clarification.json"
PIPELINE_PREREG="$REG/posttraining_pipeline_preregistration_v2.json"
PARENT_MANIFEST="$JOINT/v303_v301_batched_gates_seed1482_20260821/release_registration.json"
PARENT_RUNTIME="$BASE/pipeline/wam_pipeline/v301_batched_terminal_frame_mirror_runtime.py"
FREEZE_TOOL="$P/freeze_v307_v304_candidate.py"
FINAL_WRAPPER="$P/run_frozen_v307_v304_final128_once.sh"
THIS_LAUNCHER="$P/launch_v306_v304_public112_freeze_final.sh"
CPUSET="${TRACK2_PUBLIC_CPUSET:-16-47}"

test -f "$RUN/audit/PUBLIC_BATCH00_PASSED"
test ! -e "$RUN/audit/PUBLIC_BATCH00_REJECTED"
for path in "$CKPT" "$PREREG" "$BATCH00" "$DEV/manifest.json" \
  "$SEED_BUNDLE" "$FINAL_PREFLIGHT" "$PUBLIC_PREFLIGHT" \
  "$NUMERIC_CLARIFICATION" "$PARENT_MANIFEST" "$PARENT_RUNTIME" \
  "$FREEZE_TOOL" "$FINAL_WRAPPER" "$THIS_LAUNCHER"; do
  test -s "$path"
done
test ! -e "$RUN/audit/PUBLIC112_REJECTED"
"$PY" "$P/verify_v304_posttraining_tools.py" \
  --preregistration "$PIPELINE_PREREG" \
  --tool verify_v304_posttraining_tools.py \
  --tool launch_v306_v304_public112_freeze_final.sh \
  --tool run_strict_track2_dev_eval.sh \
  --tool audit_v276_v274_public112_gate.py \
  --tool freeze_v307_v304_candidate.py \
  --tool prepare_frozen_final128_prereg.py \
  --tool run_frozen_v307_v304_final128_once.sh \
  --tool run_strict_track2_final128_eval.sh \
  --tool summarize_frozen_final128.py \
  > "$REG/v306_tool_hash_verification.log"

restore() {
  "$PY" -m ray.scripts.scripts stop --force \
    > "$REG/ray_stop_after_v306.log" 2>&1 || true
  TRACK2_CPUSET=0-15 bash "$P/restart_v271_v274_services.sh" start \
    > "$REG/restart_v271_after_v306.log" 2>&1 || true
}
trap restore EXIT

bash "$P/restart_v301_services.sh" stop || true
bash "$P/restart_v271_v274_services.sh" stop || true
"$PY" -m ray.scripts.scripts stop --force \
  > "$REG/ray_stop_before_public112_remainder.log" 2>&1 || true

if [[ ! -f "$RUN/audit/PUBLIC112_PASSED" ]]; then
  for batch in 01 02 03 04 05 06; do
    TRACK2_DEV_SEED_ROOT="$DEV" \
    TRACK2_DEV_OUTPUT_ROOT="$OUT" \
    TRACK2_CANDIDATE_CHECKPOINT="$CKPT" \
    TRACK2_CANDIDATE_VARIANT="$VARIANT" \
    TRACK2_DEV_BATCH_FILTER="$batch" \
    TRACK2_SKIP_BASELINE=true \
      taskset -c "$CPUSET" bash "$P/run_strict_track2_dev_eval.sh" \
        > "$REG/public_batch_${batch}_eval.log" 2>&1
  done

  set +e
  "$PY" "$P/audit_v276_v274_public112_gate.py" \
    --preregistration "$PREREG" \
    --batch00-report "$BATCH00" \
    --output-root "$OUT" \
    --dev-root "$DEV" \
    --variant "$VARIANT" \
    --output "$PUBLIC112" > "$RUN/audit/public112_acceptance.log" 2>&1
  public_rc=$?
  set -e
  if (( public_rc != 0 )); then
    touch "$RUN/audit/PUBLIC112_REJECTED"
    exit "$public_rc"
  fi
  touch "$RUN/audit/PUBLIC112_PASSED"
fi
test -s "$PUBLIC112"

if [[ ! -s "$FREEZE" ]]; then
  "$PY" "$FREEZE_TOOL" \
    --checkpoint "$CKPT" \
    --training-report "$RUN/audit/p3_training_acceptance.json" \
    --training-preregistration "$REG/preregistration.json" \
    --posttraining-preregistration "$PIPELINE_PREREG" \
    --public-preregistration "$PREREG" \
    --batch00-report "$BATCH00" \
    --public112-report "$PUBLIC112" \
    --parent-manifest "$PARENT_MANIFEST" \
    --parent-runtime "$PARENT_RUNTIME" \
    --numeric-gate-clarification "$NUMERIC_CLARIFICATION" \
    --public-runner "$P/run_strict_track2_dev_eval.sh" \
    --final-runner "$P/run_strict_track2_final128_eval.sh" \
    --final-summarizer "$P/summarize_frozen_final128.py" \
    --final-preregistration-tool "$P/prepare_frozen_final128_prereg.py" \
    --final-wrapper "$FINAL_WRAPPER" \
    --public112-launcher "$THIS_LAUNCHER" \
    --final-static-preflight "$FINAL_PREFLIGHT" \
    --public-static-preflight "$PUBLIC_PREFLIGHT" \
    --variant "$VARIANT" \
    --output "$FREEZE" > "$RUN/audit/v307_freeze.log" 2>&1
fi
touch "$RUN/audit/PUBLIC112_PASSED_AND_FROZEN"

if [[ ! -s "$FINAL_PREREG" ]]; then
  "$PY" "$P/prepare_frozen_final128_prereg.py" \
    --freeze-manifest "$FREEZE" \
    --final-seed-manifest "$SEED_BUNDLE" \
    --output-root "$FINAL_OUTPUT" \
    --output "$FINAL_PREREG" \
    > "$FREEZE_DIR/final128_preregistration.log" 2>&1
fi

bash "$FINAL_WRAPPER"
