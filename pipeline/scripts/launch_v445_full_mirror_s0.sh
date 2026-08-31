#!/usr/bin/env bash
# V445 S0 only: preregister, static audit, package. Never starts S1/service/RL.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v445_full_mirror_seed1596_20260823"
RELEASE="$J/v445_v169_full_mirror_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RELEASE"
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v445_full_mirror.py" | awk '{print $1}')" = '827cd6ae387526d12742c9d627771a66ab14d58b2ca031d9b0a081c304d1037d'
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v445_v169_full_mirror_runtime.py" | awk '{print $1}')" = '4343c2a74d9e3700fc809e1ddef90cd1be3c1aca51a909efbb04d354f9aae0a0'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v445_full_mirror_static.py" | awk '{print $1}')" = 'b9887544a43b40539df07b75fe8c5606006fc91314d39ec8ecec4cfc199cebfc'
test "$(sha256sum "$ROOT/pipeline/scripts/package_v445_full_mirror_release.py" | awk '{print $1}')" = '44f1002a5a328c23cecb40f9e0ae4127a5d122ff71c50af93e088749a3f61f04'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v445_s1_offline.py" | awk '{print $1}')" = '30f96ab9c27e958c70917e0e248066abf1d246383e524532b01f01721382932f'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v445_s1_offline.py" | awk '{print $1}')" = 'e8558d2052cfa13ead3959ba26820163978411fa136acd7083564077c65817a7'

mkdir -p "$REG"
cd "$ROOT"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/prepare_v445_full_mirror.py \
  --split "$SPLIT" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/audit_v445_full_mirror_static.py \
  --runtime "$ROOT/pipeline/wam_pipeline/v445_v169_full_mirror_runtime.py" \
  --preregistration "$REG/preregistration.json" --output "$REG/s0_audit.json" >"$REG/s0_audit.log" 2>&1
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/package_v445_full_mirror_release.py \
  --preregistration "$REG/preregistration.json" --s0-audit "$REG/s0_audit.json" \
  --output "$RELEASE" >"$REG/package.log" 2>&1
echo V445_S0_COMPLETE
