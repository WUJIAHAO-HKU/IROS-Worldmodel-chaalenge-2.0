#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v263_v261_fullright_chunk8_sft2048_joint4_grip4_lr1e6_seed1463_20260819'
REG="$OFF/run_registry/$NAME"

test -d "$REG"
test ! -e "$REG/resume_after_acceptance_fix.console.log"
screen -dmS v263_resume_after_acceptance_fix bash -lc \
  "bash '$BASE/pipeline/scripts/resume_v263_after_acceptance_fix.sh' >> '$REG/resume_after_acceptance_fix.console.log' 2>&1"
sleep 2
screen -ls | grep -q v263_resume_after_acceptance_fix
echo V263_RESUME_AFTER_ACCEPTANCE_FIX_STARTED
