#!/usr/bin/env bash
# V444 public-train-only direct residual: prepare, 25 steps, kill/S0, package.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v444_direct_residual_seed1595_step25_20260823"
RELEASE="$J/v444_v169_direct_residual_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
V169="$J/v169_instruction_arm_routed_release"
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RELEASE"
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v444_direct_residual_step25.py" | awk '{print $1}')" = \
  'd72af8d2ec2438aec3c6d8d3536ea986f28f795e6ca886eb63c2ea32daaf79bd'
test "$(sha256sum "$ROOT/pipeline/scripts/train_v444_direct_residual_step25.py" | awk '{print $1}')" = \
  '51b260733f6866bec119a8312c464d670ba22a2fc0e8fca770bfadde9a7ac24c'
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v444_v169_direct_residual_runtime.py" | awk '{print $1}')" = \
  '99f8505f0aaafafc410aef2988f6581368844fcb6655e985f28f53bccee4d440'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v444_step25_killgate.py" | awk '{print $1}')" = \
  '1a59338a953b14959aea4ca3598789280ee60939918a6598f2f9a96b60495dd1'
test "$(sha256sum "$ROOT/pipeline/scripts/package_v444_direct_residual_release.py" | awk '{print $1}')" = \
  '7b090aaf97096df1177c65903bff1f4601a858100e101dfd985373993cb3903e'

mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/prepare_v444_direct_residual_step25.py \
  --split "$SPLIT" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1

cleanup() {
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export PYTHONHASHSEED=1595 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/train_v444_direct_residual_step25.py \
  --windows "$WINDOWS" --split "$SPLIT" --preregistration "$REG/preregistration.json" \
  --v169-release "$V169" --v169-library "$ROOT/artifacts" \
  --checkpoint "$REG/model_step25.pt" --report "$REG/training_report.json" \
  --device cuda >"$REG/training.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/audit_v444_step25_killgate.py \
  --checkpoint "$REG/model_step25.pt" --training-report "$REG/training_report.json" \
  --preregistration "$REG/preregistration.json" \
  --runtime "$ROOT/pipeline/wam_pipeline/v444_v169_direct_residual_runtime.py" \
  --output "$REG/s0_audit.json" >"$REG/s0_audit.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/package_v444_direct_residual_release.py \
  --checkpoint "$REG/model_step25.pt" --training-report "$REG/training_report.json" \
  --preregistration "$REG/preregistration.json" --s0-audit "$REG/s0_audit.json" \
  --output "$RELEASE" >"$REG/package.log" 2>&1
echo V444_STEP25_S0_COMPLETE
