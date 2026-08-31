#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
REG="$BASE/artifacts/strict_track2_official_20260810/run_registry/v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
SCRIPT="$BASE/pipeline/scripts/govern_v308_checkpoint_cache.py"

export PYTHONUNBUFFERED=1
exec taskset -c 0 nice -n 15 "$PY" "$SCRIPT" >>"$REG/v308_checkpoint_cache_governor.log" 2>&1
