#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
FIX="$OFF/run_registry/v227_rollout_reload_memory_fix_20260818"
export TRACK2_RETRY_NAME='v229_v228_terminal_rightsft_continue_h8_step64_seed1429_20260818'
export TRACK2_USE_EXPANDABLE_SEGMENTS=false
export TRACK2_PREVIOUS_FAILURE="$FIX/v228_gate_rejection.json"
export TRACK2_RETRY_BASE_CKPT="$OFF/runs/v228_v223_terminal_rightsft_cachefix_h8_step64_seed1426_20260818/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_64/actor/model_state_dict/full_weights.pt"
export TRACK2_RETRY_INITIAL_POLICY_NAME='v228_v223_terminal_rightsft_cachefix_h8_step64_seed1426'
export TRACK2_RETRY_ACTOR_SEED=1429
export TRACK2_RETRY_ACTOR_LR=5e-6
exec bash "$BASE/pipeline/scripts/launch_v227_terminal_right_sft_retry.sh"
