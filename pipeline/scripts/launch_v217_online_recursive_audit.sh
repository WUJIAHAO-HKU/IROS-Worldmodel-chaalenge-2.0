#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v217_v216_online_recursive_service_gate_seed1416'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
MODEL_VERSION='track2-v217-public-knn-blend-alpha070-online-gated'
URL='http://127.0.0.1:8005'
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
BASELINE="$JOINT/v212_v208_right_logit_long128_seed1411/audit/baseline"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
export PYTHONPATH="$BASE/pipeline:$P:$RLINF"

restore_parent() {
  bash "$P/restart_v217_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v209_services.sh" start >"$REG/restart_v209_after_v217.log" 2>&1 || true
}
trap restore_parent EXIT

"$PY" "$P/prepare_v217_online_recursive_service_audit.py"
ln -s "$BASELINE/public_success_long128.npz" "$RUN/audit/public_success_baseline.npz"
ln -s "$BASELINE/public_failure_long128.npz" "$RUN/audit/public_failure_baseline.npz"

screen -S wm_v209_bridge -X quit >/dev/null 2>&1 || true
screen -S wm_v209_gpu -X quit >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
  sleep 1
done
if ss -ltn | grep -qE ':(8004|18083) '; then
  echo 'v209 ports did not stop cleanly' >&2
  exit 3
fi

bash "$P/restart_v217_services.sh" start >"$RUN/v217_service_start.log" 2>&1
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

bash "$P/restart_v217_services.sh" stop
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

"$PY" "$P/summarize_v217_online_recursive.py" --run "$RUN" --registry "$REG" \
  >"$RUN/audit/v217_online_recursive_report.log" 2>&1
touch "$RUN/AUDIT_COMPLETE"
echo V217_ONLINE_RECURSIVE_AUDIT_COMPLETE
