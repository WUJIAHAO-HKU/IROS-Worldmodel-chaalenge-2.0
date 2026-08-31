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
TRAIN="$RUN/audit/p3_training_acceptance.json"
TRAIN_PREREG="$REG/preregistration.json"
WORLD_RELEASE="$JOINT/v303_v301_batched_gates_seed1482_20260821/release_registration.json"
STATIC="$RUN/audit/v304_public112_static_preflight.json"
EVAL_LOG="$OUT/$VARIANT/batch_00/launcher.log"
PREREG="$REG/public_batch00_preregistration.json"
REPORT="$RUN/audit/public_batch00_acceptance.json"
PIPELINE_PREREG="$REG/posttraining_pipeline_preregistration_v2.json"
CPUSET="${TRACK2_PUBLIC_CPUSET:-16-47}"

test -f "$RUN/audit/V304_TRAINING_ACCEPTED"
for path in "$CKPT" "$TRAIN" "$TRAIN_PREREG" "$WORLD_RELEASE" \
  "$STATIC" "$DEV/manifest.json" "$DEV/batch_00.json"; do
  test -s "$path"
done
test ! -e "$OUT/$VARIANT"
test ! -e "$PREREG"
"$PY" "$P/verify_v304_posttraining_tools.py" \
  --preregistration "$PIPELINE_PREREG" \
  --tool verify_v304_posttraining_tools.py \
  --tool prepare_v305_v304_batch00_gate.py \
  --tool launch_v305_v304_batch00_gate.sh \
  --tool run_strict_track2_dev_eval.sh \
  --tool audit_v228_public_right_gate.py \
  > "$REG/v305_tool_hash_verification.log"
grep -Fq '"passed": true' "$STATIC"
grep -Fq '"outcomes_read": false' "$STATIC"
grep -Fq '"official_submission": false' "$STATIC"

"$PY" "$P/prepare_v305_v304_batch00_gate.py" \
  --output "$PREREG" \
  --checkpoint "$CKPT" \
  --training-report "$TRAIN" \
  --training-preregistration "$TRAIN_PREREG" \
  --world-release "$WORLD_RELEASE" \
  --static-preflight "$STATIC" \
  --manifest "$DEV/manifest.json" \
  --batch00 "$DEV/batch_00.json" \
  --prepare-script "$P/prepare_v305_v304_batch00_gate.py" \
  --launcher "$P/launch_v305_v304_batch00_gate.sh" \
  --variant "$VARIANT" > "$REG/public_batch00_prepare.log"

restore() {
  "$PY" -m ray.scripts.scripts stop --force \
    > "$REG/ray_stop_after_public_batch00.log" 2>&1 || true
  TRACK2_CPUSET=0-15 bash "$P/restart_v271_v274_services.sh" start \
    > "$REG/restart_v271_after_public_batch00.log" 2>&1 || true
}
trap restore EXIT

bash "$P/restart_v301_services.sh" stop || true
bash "$P/restart_v271_v274_services.sh" stop || true
bash "$P/restart_v218_services.sh" stop || true
"$PY" -m ray.scripts.scripts stop --force \
  > "$REG/ray_stop_before_public_batch00.log" 2>&1 || true

TRACK2_DEV_SEED_ROOT="$DEV" \
TRACK2_DEV_OUTPUT_ROOT="$OUT" \
TRACK2_CANDIDATE_CHECKPOINT="$CKPT" \
TRACK2_CANDIDATE_VARIANT="$VARIANT" \
TRACK2_DEV_BATCH_FILTER=00 \
TRACK2_SKIP_BASELINE=true \
  taskset -c "$CPUSET" bash "$P/run_strict_track2_dev_eval.sh" \
    > "$REG/public_batch00_eval.log" 2>&1

set +e
"$PY" "$P/audit_v228_public_right_gate.py" "$PREREG" "$EVAL_LOG" "$REPORT" \
  > "$RUN/audit/public_batch00_acceptance.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  touch "$RUN/audit/PUBLIC_BATCH00_REJECTED"
  exit "$rc"
fi
touch "$RUN/audit/PUBLIC_BATCH00_PASSED"
echo V305_V304_PUBLIC_BATCH00_PASSED
