#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v214_public_right_knn_action_visual_diagnostic_seed1413'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
MIXED="$JOINT/v163_mixed_reward_windows"
NORMALIZATION="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407/selected_right_expert/action_normalization.npz"
BASELINE="$JOINT/v212_v208_right_logit_long128_seed1411/audit/baseline"
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

restart_services() {
  bash "$P/restart_v209_services.sh" start >"$REG/restart_v209_after_v214.log" 2>&1 || true
}
trap restart_services EXIT

"$PY" "$P/prepare_v214_public_knn_diagnostic.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"

for name in wm_v209_bridge wm_v209_gpu; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
  sleep 1
done

"$PY" "$P/build_v214_public_knn_library.py" \
  --windows "$MIXED" --split-manifest "$MIXED/split_manifest.json" \
  --normalization "$NORMALIZATION" --output "$RUN/library/public_right_knn.npz" \
  >"$RUN/library_build.log" 2>&1

"$PY" "$P/export_v214_public_knn_cache.py" \
  --baseline-cache "$RUN/audit/public_success_baseline.npz" \
  --query-windows "$PUBLIC" --library-index "$RUN/library/public_right_knn.npz" \
  --output "$RUN/audit/public_success_candidate.npz" --chunks 16 \
  --visual-weight 1.0 --action-weight 2.5 >"$RUN/audit/public_success_export.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/public_success_candidate.npz" \
  --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" \
  --reuse-baseline-report "$BASELINE/public_success_reward.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/public_success_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/public_success_reward.log" 2>&1

"$PY" "$P/export_v214_public_knn_cache.py" \
  --baseline-cache "$RUN/audit/public_failure_baseline.npz" \
  --query-windows "$ONPOLICY" --library-index "$RUN/library/public_right_knn.npz" \
  --output "$RUN/audit/public_failure_candidate.npz" --chunks 16 \
  --visual-weight 1.0 --action-weight 2.5 >"$RUN/audit/public_failure_export.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/public_failure_candidate.npz" \
  --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" \
  --reuse-baseline-report "$BASELINE/public_failure_reward.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/public_failure_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/public_failure_reward.log" 2>&1

"$PY" "$P/summarize_v214_public_knn.py" --run "$RUN" --registry "$REG" \
  >"$RUN/audit/v214_public_knn_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V214_PUBLIC_KNN_DIAGNOSTIC_COMPLETE
