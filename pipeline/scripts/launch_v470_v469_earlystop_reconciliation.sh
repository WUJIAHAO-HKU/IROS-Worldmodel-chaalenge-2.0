#!/usr/bin/env bash
# Immutable CPU-only reconciliation; no model inference, reward, policy, RL, or submission.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";V469="$J/v469_closed_form_residual_knn_seed1616_20260824";REG="$J/v470_v469_earlystop_reconciliation_seed1617_20260824"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';PREP="$ROOT/pipeline/scripts/prepare_v470_v469_earlystop_reconciliation.py";RECON="$ROOT/pipeline/scripts/audit_v470_v469_earlystop_reconciliation.py";LEGACY="$ROOT/pipeline/scripts/audit_v469_closed_form_residual_knn_s0.py";PROBE="$ROOT/pipeline/scripts/probe_v469_closed_form_residual_knn_s0.py";CONTRACT="$ROOT/pipeline/scripts/v469_closed_form_residual_knn_contract.json";SELECTION="$J/v461_endpoint200_seed1612_20260823/selection.json";DATASET="$J/v461_endpoint200_seed1612_20260823/dataset"
exec 9>/var/lock/v470_v469_earlystop_reconciliation.lock;flock -n 9||exit 73;test ! -e "$REG";mkdir -p "$REG"
test "$(sha256sum "$PREP"|awk '{print $1}')" = '40ecf5a6bf09a39b2f58e67bcf78ae73b61d08ecae3307b7dc5b2a8627dcddfe';test "$(sha256sum "$RECON"|awk '{print $1}')" = '2b4abdd00f91a64af691b14c187b9a5a60d99547d190af448dc97dfdedf612cd';test "$(sha256sum "$LEGACY"|awk '{print $1}')" = 'b9d0e133d1b38cd804caeaa79d4ce55596d7becf588dc34c05ba7fe1077fa60b';test "$(sha256sum "$PROBE"|awk '{print $1}')" = '9497cdfcaae0e8a79c65fe7cb602e8a66e63790764b5e84e02056531a12ba123'
"$RLPY" -m py_compile "$PREP" "$RECON" "$LEGACY"
env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 timeout 30s "$RLPY" "$PREP" --v469-reg "$V469" --legacy-auditor "$LEGACY" --reconciler "$RECON" --launcher "$0" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
set +e
env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0-5 timeout 90s "$RLPY" "$LEGACY" --preregistration "$V469/preregistration.json" --contract "$CONTRACT" --probe "$PROBE" --selection "$SELECTION" --dataset "$DATASET" --result-dir "$V469/result" --output "$REG/legacy_recomputed.json" >"$REG/legacy_recompute.log" 2>&1
CODE=$?
set -e
test "$CODE" -eq 3;test -s "$REG/legacy_recomputed.json"
env PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 timeout 30s "$RLPY" "$RECON" --preregistration "$REG/preregistration.json" --legacy-recomputed "$REG/legacy_recomputed.json" --output "$REG/reconciliation_receipt.json" >"$REG/reconcile.log" 2>&1
echo V470_V469_EARLYSTOP_RECONCILIATION_COMPLETE
