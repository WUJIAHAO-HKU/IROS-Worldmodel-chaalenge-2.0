#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge';OFF="$BASE/artifacts/strict_track2_official_20260810";JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810";NAME='v243b_public_clean_successor_reward_map_seed1445_20260819';RUN="$JOINT/$NAME";P="$BASE/pipeline/scripts";GO1='/root/miniconda3/envs/go1/bin/python';RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python';RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
"$GO1" "$P/prepare_v243_clean_reward_map.py"
cd /tmp
PYTHONPATH="$BASE/pipeline:$RLINF" TOKENIZERS_PARALLELISM=false "$RLPY" "$P/map_v243_clean_successor_rewards.py" \
 --library "$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
 --audit-dir "$OFF/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit" \
 --reward-checkpoint "$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" --t5-model "$BASE/artifacts/official_resources/reward_model/t5-base" \
 --batch-size 12 --output "$RUN/audit/reward_map_report.json" --index-output "$RUN/audit/clean_successor_rewards.npz" >"$RUN/audit/reward_map.log" 2>&1
touch "$RUN/V243_ANALYSIS_COMPLETE"
