#!/usr/bin/env bash
# CPU-only immutable-cache S0; no GPU/service stop/reward/policy/RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";REG="$J/v471_group_context_prototype_knn_seed1616_20260824";RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';PREP="$ROOT/pipeline/scripts/prepare_v471_group_context_prototype_knn.py";PROBE="$ROOT/pipeline/scripts/probe_v471_group_context_prototype_knn_s0.py";AUDIT="$ROOT/pipeline/scripts/audit_v471_group_context_prototype_knn_s0.py";CONTRACT="$ROOT/pipeline/scripts/v471_group_context_prototype_knn_contract.json";V461="$J/v461_endpoint200_seed1612_20260823";SELECTION="$V461/selection.json";DATASET="$V461/dataset"
exec 9>/var/lock/v471_group_context_prototype_knn.lock;flock -n 9||exit 73;test ! -e "$REG";mkdir -p "$REG"
test "$(sha256sum "$CONTRACT"|awk '{print $1}')" = 'cec9a46deb42949f4f12a5d9f81821ab8eb22cb82cacdb90794ed10c5574978a';test "$(sha256sum "$PREP"|awk '{print $1}')" = 'b73366bbbfcd8583401a6071aecb00255a055207153555b504cd9de666a3b783';test "$(sha256sum "$PROBE"|awk '{print $1}')" = '93af14f2f2d189aa9aa0775189313c391f375098fe30824b3cfcbf37e9ba453c';test "$(sha256sum "$AUDIT"|awk '{print $1}')" = '045bcfcd55e06f0948f677fab4814dd145f0a878b90812176fb3caf871c4e8bb';"$RLPY" -m py_compile "$PREP" "$PROBE" "$AUDIT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONHASHSEED=0
START=$(date +%s)
remain(){ local r=$((300-$(date +%s)+START));((r>0))||return 124;printf '%s' "$r";}
R=$(remain);taskset -c 0-11 timeout "${R}s" "$RLPY" "$PREP" --contract "$CONTRACT" --probe "$PROBE" --auditor "$AUDIT" --launcher "$0" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
R=$(remain);set +e;taskset -c 0-11 timeout "${R}s" "$RLPY" "$PROBE" --preregistration "$REG/preregistration.json" --contract "$CONTRACT" --selection "$SELECTION" --dataset "$DATASET" --output-dir "$REG/result" >"$REG/probe.log" 2>&1;CODE=$?;set -e
R=$(remain);taskset -c 0-11 timeout "${R}s" "$RLPY" "$AUDIT" --preregistration "$REG/preregistration.json" --contract "$CONTRACT" --probe "$PROBE" --selection "$SELECTION" --dataset "$DATASET" --result-dir "$REG/result" --output "$REG/audit_receipt.json" >"$REG/audit.log" 2>&1
if [[ "$CODE" -ne 0 ]];then exit "$CODE";fi
echo V471_GROUP_CONTEXT_PROTOTYPE_KNN_S0_PASS
