#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v215_public_right_knn_visual_weight_sweep_seed1414'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
V214="$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
BASELINE="$JOINT/v212_v208_right_logit_long128_seed1411/audit/baseline"
LIBRARY="$V214/library/public_right_knn.npz"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

restart_services() {
  bash "$P/restart_v209_services.sh" start >"$REG/restart_v209_after_v215.log" 2>&1 || true
}
trap restart_services EXIT

"$PY" "$P/prepare_v215_knn_visual_weight_sweep.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"
for name in wm_v209_bridge wm_v209_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 30); do if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi; sleep 1; done

for spec in visual05:5 visual10:10 visual25:25; do
  tag=${spec%%:*}
  visual=${spec##*:}
  "$PY" "$P/export_v214_public_knn_cache.py" \
    --baseline-cache "$RUN/audit/public_success_baseline.npz" --query-windows "$PUBLIC" \
    --library-index "$LIBRARY" --output "$RUN/audit/${tag}_public_success_candidate.npz" \
    --chunks 16 --visual-weight "$visual" --action-weight 2.5 >"$RUN/audit/${tag}_public_success_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$RUN/audit/${tag}_public_success_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" \
    --reuse-baseline-report "$BASELINE/public_success_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$RUN/audit/${tag}_public_success_reward.json" --batch-size 32 --device cuda \
    >"$RUN/audit/${tag}_public_success_reward.log" 2>&1
  "$PY" "$P/export_v214_public_knn_cache.py" \
    --baseline-cache "$RUN/audit/public_failure_baseline.npz" --query-windows "$ONPOLICY" \
    --library-index "$LIBRARY" --output "$RUN/audit/${tag}_public_failure_candidate.npz" \
    --chunks 16 --visual-weight "$visual" --action-weight 2.5 >"$RUN/audit/${tag}_public_failure_export.log" 2>&1
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$RUN/audit/${tag}_public_failure_candidate.npz" \
    --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" \
    --reuse-baseline-report "$BASELINE/public_failure_reward.json" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$RUN/audit/${tag}_public_failure_reward.json" --batch-size 32 --device cuda \
    >"$RUN/audit/${tag}_public_failure_reward.log" 2>&1
  touch "$RUN/audit/${tag}_COMPLETE"
done

"$PY" "$P/summarize_v215_knn_sweep.py" --run "$RUN" --registry "$REG" >"$RUN/audit/v215_knn_visual_sweep_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V215_KNN_VISUAL_SWEEP_COMPLETE
