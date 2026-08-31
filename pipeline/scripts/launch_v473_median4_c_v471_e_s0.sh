#!/usr/bin/env bash
# CPU-only immutable-cache v473 S0; no GPU/service stop/reward/policy/RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";REG="$J/v473_median4_c_v471_e_seed1616_20260824";RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';S="$ROOT/pipeline/scripts";PREP="$S/prepare_v473_median4_c_v471_e.py";PROBE="$S/probe_v473_median4_c_v471_e_s0.py";AUDIT="$S/audit_v473_median4_c_v471_e_s0.py";CONTRACT="$S/v473_median4_c_v471_e_contract.json";V461="$J/v461_endpoint200_seed1612_20260823";SELECTION="$V461/selection.json";DATASET="$V461/dataset"
exec 9>/var/lock/v473_median4_c_v471_e.lock;flock -n 9||exit 73;test ! -e "$REG";mkdir -p "$REG"
test "$(sha256sum "$CONTRACT"|awk '{print $1}')" = '06d7869f1beef1c67afe1a64d0de14d992d7b0993f5f7d3739debd9e685b5803';test "$(sha256sum "$PREP"|awk '{print $1}')" = 'c3c4116b0b7d42edfe960bcc96416a0519225b050d9f6cce0484899f2f3314b2';test "$(sha256sum "$PROBE"|awk '{print $1}')" = '5d0b700218b088b1dbe9a7126fca688a13ea7446f6011674c0f157965a5e2540';test "$(sha256sum "$AUDIT"|awk '{print $1}')" = '0bc760a71e5514330e17386b287f54aee85f926461c8f4da52b4f846c2af9a4e';"$RLPY" -m py_compile "$PREP" "$PROBE" "$AUDIT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONHASHSEED=0;START=$(date +%s);remain(){ local r=$((300-$(date +%s)+START));((r>0))||return 124;printf '%s' "$r";}
R=$(remain);taskset -c 0-11 timeout "${R}s" "$RLPY" "$PREP" --contract "$CONTRACT" --probe "$PROBE" --auditor "$AUDIT" --launcher "$0" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
R=$(remain);set +e;taskset -c 0-11 timeout "${R}s" "$RLPY" "$PROBE" --preregistration "$REG/preregistration.json" --contract "$CONTRACT" --selection "$SELECTION" --dataset "$DATASET" --output-dir "$REG/result" >"$REG/probe.log" 2>&1;CODE=$?;set -e
R=$(remain);taskset -c 0-11 timeout "${R}s" "$RLPY" "$AUDIT" --preregistration "$REG/preregistration.json" --contract "$CONTRACT" --probe "$PROBE" --selection "$SELECTION" --dataset "$DATASET" --result-dir "$REG/result" --output "$REG/audit_receipt.json" >"$REG/audit.log" 2>&1
if [[ "$CODE" -ne 0 ]];then exit "$CODE";fi
echo V473_MEDIAN4_C_V471_E_S0_PASS
