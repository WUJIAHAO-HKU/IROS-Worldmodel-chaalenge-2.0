#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
RUN="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405"
P="$BASE/pipeline/scripts"
PY='/root/miniconda3/envs/go1/bin/python'

restart_v202() {
  "$JOINT/v202_v201_public_terminal_reward_calibration_seed1402/restart_services_after_reboot.sh" start \
    > "$RUN/restart_v202_after_v205.log" 2>&1 || true
}
trap restart_v202 EXIT

screen -S wm_v202_bridge -X quit 2>/dev/null || true
screen -S wm_v202_gpu -X quit 2>/dev/null || true
screen -S wm_v196_bridge -X quit 2>/dev/null || true
screen -S wm_v196_gpu -X quit 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then break; fi
  sleep 1
done
if ss -ltn | grep -qE ':(8004|18083) '; then
  echo 'world-model ports did not stop cleanly' >&2
  exit 3
fi

bash "$P/run_v205_baseline_audit.sh"
bash "$P/run_v205_training.sh"
bash "$P/run_v205_candidate_audit.sh"
"$PY" "$P/select_v205_right_terminal_parent.py" \
  > "$RUN/audit/selection.log" 2>&1
touch "$RUN/PIPELINE_COMPLETE"
cat "$RUN/audit/selection.log"
