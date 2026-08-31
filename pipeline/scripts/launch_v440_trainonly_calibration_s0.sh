#!/usr/bin/env bash
# Public-train-only v440 calibration, S0 audit and diagnostic packaging.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v440_trainonly_aligned_projection_seed1588_20260823"
RELEASE="$J/v440_v169_trainonly_aligned_projection_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
V169="$J/v169_instruction_arm_routed_release"
V436="$J/v436_v432_step25_parent_diagnostic_release"
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RELEASE"
test -s "$ROOT/pipeline/wam_pipeline/v15_texture_reprojection.py"
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v15_texture_reprojection.py" | awk '{print $1}')" = \
  'bef1f32c62e4eacd11171374302de83802344f59968aeb445aa8212d6f017c70'
mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline" taskset -c 0-5 "$GO1" \
  pipeline/scripts/prepare_v440_trainonly_aligned_projection.py \
  --split "$SPLIT" --output "$REG/preregistration.json" \
  >"$REG/prepare.log" 2>&1

cleanup() {
  bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start \
    >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT
for name in wm_v218_bridge wm_v218_gpu; do
  screen -S "$name" -X quit >/dev/null 2>&1 || true
done
for _ in $(seq 1 60); do
  ! ss -ltn | grep -qE ':(8005|18084) ' && break
  sleep 1
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
PYTHONPATH="$ROOT/pipeline" taskset -c 0-11 "$GO1" \
  pipeline/scripts/calibrate_v440_trainonly_rgb_projection.py \
  --windows "$WINDOWS" --split "$SPLIT" \
  --v169-release "$V169" --v169-library "$ROOT/artifacts" \
  --v436-release "$V436" --preregistration "$REG/preregistration.json" \
  --output "$REG/alignment_index.json" --device cuda --inference-batch-size 4 \
  >"$REG/calibration.log" 2>&1

PYTHONPATH="$ROOT/pipeline" taskset -c 0-5 "$GO1" \
  pipeline/scripts/audit_v440_static_contract.py \
  --alignment-index "$REG/alignment_index.json" \
  --preregistration "$REG/preregistration.json" --split "$SPLIT" \
  --runtime "$ROOT/pipeline/wam_pipeline/v440_v169_trainonly_aligned_projection_runtime.py" \
  --output "$REG/s0_audit.json" >"$REG/s0_audit.log" 2>&1

PYTHONPATH="$ROOT/pipeline" taskset -c 0-5 "$GO1" \
  pipeline/scripts/package_v440_trainonly_aligned_release.py \
  --preregistration "$REG/preregistration.json" \
  --alignment-index "$REG/alignment_index.json" --s0-audit "$REG/s0_audit.json" \
  --output "$RELEASE" >"$REG/package.log" 2>&1
echo V440_S0_COMPLETE
