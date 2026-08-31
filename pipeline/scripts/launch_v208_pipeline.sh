#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
RUN="$BASE/artifacts/strict_track2_joint_augmentation_20260810/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'

cleanup() {
  /root/autodl-tmp/conda_envs/rlinf_track2/bin/ray stop --force >/dev/null 2>&1 || true
  cd "$BASE"
  bash "$P/restart_v206_services.sh" >"$RUN/restart_v206_after_v208.log" 2>&1 || true
}
trap cleanup EXIT

cd "$BASE"
"$PY" "$P/prepare_v208_mixed_right_contrast.py"
screen -S wm_v206_bridge -X quit 2>/dev/null || true
screen -S wm_v206_gpu -X quit 2>/dev/null || true
/root/autodl-tmp/conda_envs/rlinf_track2/bin/ray stop --force >/dev/null 2>&1 || true

bash "$P/run_v208_baseline_audit.sh"
bash "$P/run_v208_training.sh"
bash "$P/run_v208_candidate_audit.sh"
"$PY" "$P/select_v208_mixed_right_contrast.py"
echo V208_PIPELINE_COMPLETE
