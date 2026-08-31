#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
REG="$BASE/artifacts/strict_track2_official_20260810/run_registry/v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
export PYTHONUNBUFFERED=1
exec taskset -c 22 nice -n 15 /root/autodl-tmp/conda_envs/rlinf_track2/bin/python \
  "$BASE/pipeline/scripts/govern_v318_checkpoint_cache.py" \
  >>"$REG/v318_checkpoint_cache_governor.log" 2>&1
