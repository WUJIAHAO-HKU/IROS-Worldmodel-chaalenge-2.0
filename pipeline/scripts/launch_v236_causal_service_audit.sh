#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v236_v235c_causal_right_close_service_seed1435_20260818'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
MODEL_VERSION='track2-v236-public-knn-alpha070-causal-right-close'
URL='http://127.0.0.1:8005'
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
BASELINE="$JOINT/v212_v208_right_logit_long128_seed1411/audit/baseline"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

restore_v218() {
  bash "$P/restart_v236_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restart_v218_after_v236.log" 2>&1 || true
}
trap restore_v218 EXIT

"$PY" "$P/prepare_v236_causal_service_audit.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"

bash "$P/restart_v236_services.sh" start >"$RUN/v236_service_start.log" 2>&1
"$PY" "$P/strict_service_acceptance.py" \
  --base-url "$URL" --token-file "$RUN/local_dev_token.txt" \
  --model-version "$MODEL_VERSION" --output "$RUN/audit/service_acceptance.json" \
  >"$RUN/audit/service_acceptance.log" 2>&1

"$PY" "$P/export_v217_service_recursive_cache.py" \
  --baseline-cache "$RUN/audit/public_success_baseline.npz" \
  --query-windows "$PUBLIC" --url "$URL" --token local-dev-token \
  --model-version "$MODEL_VERSION" --output "$RUN/audit/public_success_candidate.npz" \
  --chunks 16 --batch-size 8 >"$RUN/audit/public_success_export.log" 2>&1
"$PY" "$P/export_v217_service_recursive_cache.py" \
  --baseline-cache "$RUN/audit/public_failure_baseline.npz" \
  --query-windows "$ONPOLICY" --url "$URL" --token local-dev-token \
  --model-version "$MODEL_VERSION" --output "$RUN/audit/public_failure_candidate.npz" \
  --chunks 16 --batch-size 8 >"$RUN/audit/public_failure_export.log" 2>&1

bash "$P/restart_v236_services.sh" stop
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -q ':8005 '; then break; fi
  sleep 1
done

"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/public_success_candidate.npz" \
  --reuse-baseline-cache "$RUN/audit/public_success_baseline.npz" \
  --reuse-baseline-report "$BASELINE/public_success_reward.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/public_success_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/public_success_reward.log" 2>&1
"$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
  --cache "$RUN/audit/public_failure_candidate.npz" \
  --reuse-baseline-cache "$RUN/audit/public_failure_baseline.npz" \
  --reuse-baseline-report "$BASELINE/public_failure_reward.json" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$RUN/audit/public_failure_reward.json" --batch-size 32 --device cuda \
  >"$RUN/audit/public_failure_reward.log" 2>&1

"$PY" "$P/summarize_v236_causal_service.py" --run "$RUN" --registry "$REG" \
  >"$RUN/audit/v236_causal_service_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V236_CAUSAL_SERVICE_AUDIT_COMPLETE
