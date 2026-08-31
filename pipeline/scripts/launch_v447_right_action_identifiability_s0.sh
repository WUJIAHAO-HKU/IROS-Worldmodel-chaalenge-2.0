#!/usr/bin/env bash
# V447 train-only, CPU-only action-identifiability S0. No reward/model/service/RL.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v447_right_action_identifiability_seed1600_20260823"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
GO1='/root/miniconda3/envs/go1/bin/python'

PREPARE_SHA='13f5dec725df3cfb45d750d3c98bee0a5ea68c13d52653bb51884ba4c985b4fb'
PROBE_SHA='583329c29dd7feb2d6fa7f0366d07a1c6e94b2110951f0c10d199ff0b1d43761'
AUDIT_SHA='c70eb2c93edc1befac6425e19f8bf6c63b02e7d06344e51341e89f13482aebeb'

test ! -e "$REG"
test -d "$WINDOWS"
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v447_right_action_identifiability.py" | awk '{print $1}')" = "$PREPARE_SHA"
test "$(sha256sum "$ROOT/pipeline/scripts/probe_v447_right_action_identifiability.py" | awk '{print $1}')" = "$PROBE_SHA"
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v447_right_action_identifiability.py" | awk '{print $1}')" = "$AUDIT_SHA"

mkdir -p "$REG"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=''
export PYTHONHASHSEED=1600
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/prepare_v447_right_action_identifiability.py \
  --split "$SPLIT" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/probe_v447_right_action_identifiability.py \
  --windows "$WINDOWS" --split "$SPLIT" \
  --preregistration "$REG/preregistration.json" --output "$REG/probe_report.json" \
  >"$REG/probe.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/audit_v447_right_action_identifiability.py \
  --preregistration "$REG/preregistration.json" --report "$REG/probe_report.json" \
  --probe "$ROOT/pipeline/scripts/probe_v447_right_action_identifiability.py" \
  --output "$REG/s0_audit.json" >"$REG/s0_audit.log" 2>&1

echo V447_ACTION_IDENTIFIABILITY_S0_PASSED
