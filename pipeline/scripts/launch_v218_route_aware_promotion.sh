#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v218_v217_route_aware_service_promotion_seed1417'
RUN="$JOINT/$NAME"
REG="$OFF/run_registry/$NAME"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'
MODEL_VERSION='track2-v218-public-knn-blend-alpha070-route-aware'
BASELINE="$JOINT/v212_v208_right_logit_long128_seed1411/audit/baseline/public_success_long128.npz"
PUBLIC="$BASE/artifacts/adjust_bottle_windows_full"
export PYTHONPATH="$BASE/pipeline:$P"

passed=0
cleanup() {
  if [[ "$passed" != 1 ]]; then
    bash "$P/restart_v218_services.sh" stop >/dev/null 2>&1 || true
    bash "$P/restart_v209_services.sh" start >"$REG/restart_v209_after_v218_failure.log" 2>&1 || true
  fi
}
trap cleanup EXIT

"$PY" "$P/prepare_v218_route_aware_promotion.py"
for name in wm_v209_bridge wm_v209_gpu; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
  sleep 1
done
if ss -ltn | grep -qE ':(8004|18083) '; then
  echo 'v209 ports did not stop cleanly' >&2
  exit 3
fi
bash "$P/restart_v218_services.sh" start >"$RUN/v218_service_start.log" 2>&1
"$PY" "$P/audit_v218_real_context_determinism.py" \
  --baseline-cache "$BASELINE" --windows "$PUBLIC" \
  --url http://127.0.0.1:8005 --token local-dev-token \
  --model-version "$MODEL_VERSION" --output "$RUN/audit/real_context_determinism.json" \
  >"$RUN/audit/real_context_determinism.log" 2>&1
"$PY" "$P/summarize_v218_route_aware_promotion.py" --run "$RUN" --registry "$REG" \
  >"$RUN/audit/v218_route_aware_promotion_report.log" 2>&1
test -f "$RUN/V218_SERVICE_PROMOTED"
touch "$RUN/AUDIT_COMPLETE"
passed=1
echo V218_ROUTE_AWARE_SERVICE_PROMOTED
