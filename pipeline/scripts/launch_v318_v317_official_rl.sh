#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
P="$BASE/pipeline/scripts"
J="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
AUTH="$J/v317_v315_native_batch_causal_gate_seed1490_20260822/audit/expensive_rl_authorization.json"
EXPECTED_AUTH_SHA='a8dec8fdb42b8a2ee030c3127787bff0ca6fabefe49041c3e3a36d436887fa7f'
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

test "$(sha256sum "$AUTH" | awk '{print $1}')" = "$EXPECTED_AUTH_SHA"
"$PY" - "$AUTH" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
assert value.get("passed") is True and value.get("launch_permission") is True
assert all(value.get("checks", {}).values())
PY

export TRACK2_LAUNCH_NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
export TRACK2_LAUNCH_RELEASE="$J/v317_v315_native_batch_causal_gate_seed1490_20260822"
export TRACK2_LAUNCH_MODEL_VERSION='track2-v317-batched-sparse-failure-terminal-v315'
export TRACK2_LAUNCH_PREPARE_SCRIPT='prepare_v318_v317_authorized_official_rl.py'
export TRACK2_LAUNCH_SERVICE_SCRIPT='restart_v317_services.sh'
export TRACK2_LAUNCH_ACCEPT_MARKER='V318_TRAINING_ACCEPTED'
export TRACK2_SERVICE_CPUSET='0-5'
export TRACK2_RL_CPUSET='6-21'

screen -S v308_checkpoint_cache_governor -X quit >/dev/null 2>&1 || true
exec bash "$P/launch_v300_v295_official_rl.sh"
