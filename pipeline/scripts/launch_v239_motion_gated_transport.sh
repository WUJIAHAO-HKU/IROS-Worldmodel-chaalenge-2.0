#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v239_public_motion_gated_transport_seed1438_20260819'
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"

"$PY" "$ROOT/pipeline/scripts/prepare_v239_motion_gated_transport.py" \
  >"$OFF/run_registry/$NAME.prepare.log" 2>&1
mv "$OFF/run_registry/$NAME.prepare.log" "$REG/prepare.log"
cd /tmp
PYTHONPATH="$ROOT/pipeline:$RLINF" "$PY" "$ROOT/pipeline/scripts/analyze_v239_motion_gated_transport.py" \
  --audit-dir "$OFF/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit" \
  --library "$JOINT/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz" \
  --output "$REG/analysis_report.json" >"$REG/analysis.log" 2>&1
touch "$REG/AUDIT_COMPLETE"
echo V239_MOTION_GATED_TRANSPORT_COMPLETE
