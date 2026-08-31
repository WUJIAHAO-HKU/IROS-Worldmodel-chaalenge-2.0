#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
FIX="$BASE/artifacts/strict_track2_official_20260810/run_registry/v227_rollout_reload_memory_fix_20260818"
export TRACK2_RETRY_NAME='v228_v223_terminal_rightsft_cachefix_h8_step64_seed1426_20260818'
export TRACK2_USE_EXPANDABLE_SEGMENTS=false
export TRACK2_PREVIOUS_FAILURE="$FIX/v227_ipc_failure.json"
exec bash "$BASE/pipeline/scripts/launch_v227_terminal_right_sft_retry.sh"
