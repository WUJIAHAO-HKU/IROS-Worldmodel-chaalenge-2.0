#!/usr/bin/env bash
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN="$ROOT/artifacts/strict_track2_joint_augmentation_20260810/v375_bounded_cartesian_phase_pilot_seed1538_20260823"
PY=/root/miniconda3/envs/go1/bin/python
P="$ROOT/pipeline/scripts"
restore() {
  bash "$P/restart_v375_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$RUN/restore_v218.log" 2>&1 || true
}
trap restore EXIT
test -s "$RUN/audit/runtime_equivalence.json"
test "$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["passed"])' "$RUN/audit/runtime_equivalence.json")" = True
bash "$P/restart_v375_services.sh" start >"$RUN/start_service.log" 2>&1
printf '%s\n' local-dev-token >"$RUN/local_dev_token.txt"
PYTHONPATH="$ROOT/pipeline" "$PY" "$P/strict_service_acceptance.py" \
  --base-url http://127.0.0.1:8005 --token-file "$RUN/local_dev_token.txt" \
  --model-version track2-v375-bounded-cartesian-phase-r16-p1 \
  --output "$RUN/audit/service_acceptance.json" >"$RUN/audit/service_acceptance.log" 2>&1
touch "$RUN/SERVICE_ACCEPTANCE_COMPLETE"
printf 'V375_SERVICE_ACCEPTANCE_COMPLETE\n'
