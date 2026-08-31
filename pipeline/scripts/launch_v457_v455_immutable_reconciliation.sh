#!/usr/bin/env bash
# Read-only v455 audit reconciliation. No simulator, reward, model, policy, or RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
V455="$J/v455_postclose_endpoint_pilot_seed1608_20260823"
REG="$J/v457_v455_immutable_reconciliation_seed1610_20260823"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
PREP="$ROOT/pipeline/scripts/prepare_v457_v455_immutable_reconciliation.py"
AUDIT="$ROOT/pipeline/scripts/audit_v457_v455_immutable_reconciliation.py"
test ! -e "$REG"
test -f "$V455/preregistration.json"
test -f "$V455/dataset/generation_report.json"
test -f "$V455/audit.json"
test "$(find "$V455/dataset" -maxdepth 1 -name 'episode*_start*.npz' -type f | wc -l)" -eq 4
test "$(readlink -f "$("$RLPY" -c 'import sys; print(sys.executable)')")" = "$(readlink -f "$RLPY")"
test "$(sha256sum "$PREP" | awk '{print $1}')" = 'df8b3097acd9dfe84f08dd448b3df02ae201c46fff6d51a786ec2d1479b18f4f'
test "$(sha256sum "$AUDIT" | awk '{print $1}')" = '496e64dd8870c4e38ae1df33010a6086781267df83d7644f3abe19bbe54eb888'
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
mkdir -p "$REG"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
taskset -c 0-5 "$RLPY" "$PREP" \
  --v455-registry "$V455" \
  --generator "$ROOT/pipeline/scripts/generate_v455_postclose_endpoint_pilot.py" \
  --auditor "$ROOT/pipeline/scripts/audit_v455_postclose_endpoint_pilot.py" \
  --resize-source "$ROOT/pipeline/wam_pipeline/data.py" \
  --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
timeout --signal=TERM --kill-after=5s 60s taskset -c 0-5 "$RLPY" "$AUDIT" \
  --preregistration "$REG/preregistration.json" \
  --recomputed-output "$REG/recomputed_v455_audit.json" \
  --output "$REG/receipt.json" >"$REG/audit.log" 2>&1
"$RLPY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True and p["endpoint_parent_data_authorized"] is True and p["guards"]["rl_authorized"] is False' "$REG/receipt.json"
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
echo V457_V455_IMMUTABLE_RECONCILIATION_COMPLETE
