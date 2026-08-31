#!/usr/bin/env bash
set -euo pipefail

base='/root/autodl-tmp/IROS_WAM_2.0 challenge'
off="$base/artifacts/strict_track2_official_20260810"
name='v204_v202_effectivekl_h200_r2_step8_lr1e5_beta005_seed1404_20260818'
run="$off/runs/$name"
status="$run/audit/post_training_watcher_status.json"
py='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

while [[ ! -s "$status" ]]; do sleep 60; done
"$py" - "$status" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["passed"] is True
PY
screen -dmS v204_public_stage_a bash -lc \
  "'$base/pipeline/scripts/run_v204_public_stage_a.sh' > '$run/audit/public_policy_stage_a_launcher.log' 2>&1"
