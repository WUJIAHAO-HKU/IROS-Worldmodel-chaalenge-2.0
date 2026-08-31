#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge';OFF="$BASE/artifacts/strict_track2_official_20260810";JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810";NAME='v244_public_clean_progressive_proxy_seed1446_20260819';P="$BASE/pipeline/scripts";PY='/root/miniconda3/envs/go1/bin/python'
"$PY" "$P/prepare_v244_progressive_proxy.py"
cd "$BASE"
PYTHONPATH="$BASE/pipeline" "$PY" "$P/analyze_v244_progressive_successor_proxy.py" \
 --library "$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
 --reward-map "$JOINT/v243b_public_clean_successor_reward_map_seed1445_20260819/audit/clean_successor_rewards.npz" \
 --details "$JOINT/v242b_public_reward_blend_scale_sweep_seed1443_20260819/audit/scale_sweep_details.npz" \
 --audit-dir "$OFF/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit" \
 --output "$OFF/run_registry/$NAME/analysis_report.json" >"$OFF/run_registry/$NAME/analysis.log" 2>&1
touch "$OFF/run_registry/$NAME/V244_ANALYSIS_COMPLETE"
