#!/usr/bin/env bash
set -euo pipefail
export TRACK2_ROLLOUTONLY_NAME='v363_v169_v326_matched_rolloutonly32_seed1497_20260822'
export TRACK2_SERVICE_WRAPPER='restart_v326_services.sh'
export TRACK2_MODEL_VERSION='track2-v326-blended-phase-terminal-v317'
export TRACK2_BRIDGE_PORT=18084
export TRACK2_ACTOR_SEED=1497
export TRACK2_EVAL_GROUP_SIZE=4
exec bash '/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/run_v169_worldmodel_rolloutonly32.sh'
