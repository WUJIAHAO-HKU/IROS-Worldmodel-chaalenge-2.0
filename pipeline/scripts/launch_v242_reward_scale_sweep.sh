#!/usr/bin/env bash
set -euo pipefail
BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v242b_public_reward_blend_scale_sweep_seed1443_20260819'
RUN="$JOINT/$NAME"
P="$BASE/pipeline/scripts"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"

restore() {
  bash "$P/restart_v241_services.sh" stop >/dev/null 2>&1 || true
  screen -S wm_v209_bridge -X quit >/dev/null 2>&1 || true
  screen -S wm_v209_gpu -X quit >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$RUN/restart_v218.log" 2>&1 || true
}
trap restore EXIT

"$GO1" "$P/prepare_v242_reward_scale_sweep.py"
bash "$P/restart_v209_services.sh" start >"$RUN/start_v209.log" 2>&1
bash "$P/restart_v241_services.sh" start >"$RUN/start_v241.log" 2>&1
cd /tmp
PYTHONPATH="$BASE/pipeline:$RLINF" TOKENIZERS_PARALLELISM=false "$RLPY" "$P/sweep_v242_service_blend_scale.py" \
  --audit-dir "$OFF/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit" \
  --library "$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --parent-url http://127.0.0.1:8004 --parent-version track2-v209-public-arm-routed-v202-left-v208-right-selected \
  --v241-url http://127.0.0.1:8005 --v241-version track2-v241-alignment-gated-right-transport \
  --token local-dev-token \
  --reward-checkpoint "$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt" \
  --t5-model "$BASE/artifacts/official_resources/reward_model/t5-base" \
  --output "$RUN/audit/scale_sweep_report.json" --details "$RUN/audit/scale_sweep_details.npz" \
  >"$RUN/audit/scale_sweep.log" 2>&1
touch "$RUN/V242_ANALYSIS_COMPLETE"
