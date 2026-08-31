#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
NAME='v411_v169step5_v409_oneupdate_h200_r4_lr1e5_seed1569_20260823'
RUN="/dev/shm/$NAME"
REG="$O/run_registry/$NAME"
V169="$O/runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
REFERENCE="$O/immutable_reference_policy_state/pi05_official_rank0.pt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
EXP='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
CHECKPOINT="$RUN/$EXP/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test -s "$REG/preregistration.json"
test ! -e "$RUN"
test "$(sha256sum "$V169" | awk '{print $1}')" = '41f0bee86472d22cbb6ab6a7d0060f08f0ba0aad93d260582a5f7f078bca776b'
free_shm=$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')
(( free_shm >= 30000000000 )) || { echo "V411_INSUFFICIENT_SHM=$free_shm" >&2; exit 4; }

cleanup() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  TRACK2_SERVICE_REG="$REG" bash "$P/restart_v409_services.sh" stop >/dev/null 2>&1 || true
  bash "$P/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true
}
trap cleanup EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 TRACK2_WAM_RELEASE_CUDA_CACHE=1 \
  bash "$P/restart_v409_services.sh" start >"$REG/restart_v409_before_training.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q 'track2-v409-v407-half-contracted-progressive'

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION='track2-v409-v407-half-contracted-progressive'
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$J/v409_half_contracted_progressive_seed1567_20260823/release_registration.json"
export TRACK2_MAX_STEPS=1 TRACK2_SAVE_INTERVAL=1 TRACK2_KEEP_LAST_CHECKPOINTS=1
export TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1569 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=1e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200
export TRACK2_ROLLOUT_EPOCH=4 TRACK2_ACTOR_GLOBAL_BATCH_SIZE=3200
export TRACK2_CKPT_PATH="$V169" TRACK2_START_STEP=0
export TRACK2_REFERENCE_STATE_STORAGE=disk TRACK2_REFERENCE_STATE_PATH_OVERRIDE="$REFERENCE"
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true
export TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true TRACK2_CATCH_SYSTEM_FAILURE=0
export TRACK2_ACTOR_OFFLOAD=true TRACK2_ENABLE_SFT_CO_TRAIN=false
export TOKENIZERS_PARALLELISM=false TRACK2_CPUSET='6-21'
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TRACK2_RAY_SYSTEM_CONFIG_JSON='{"grpc_client_keepalive_timeout_ms":1800000,"grpc_keepalive_timeout_ms":1800000,"health_check_timeout_ms":1800000,"health_check_failure_threshold":10}'

bash "$P/run_strict_track2_conservative_kl.sh" >"$REG/runner.screen.log" 2>&1
test -s "$CHECKPOINT"
"$PY" -c 'import sys,zipfile;raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$CHECKPOINT"
"$PY" "$P/audit_strict_track2_kl_smoke.py" \
  --run "$RUN" --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/training_acceptance.json" >"$REG/training_acceptance.log" 2>&1
cp "$RUN/audit/training_acceptance.json" "$REG/training_acceptance.json"
"$PY" - "$REG/training_acceptance.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload.get("passed") is True
PY
sha256sum "$CHECKPOINT" >"$REG/volatile_checkpoint.sha256"
printf '%s\n' "$CHECKPOINT" >"$REG/volatile_checkpoint.path"
touch "$REG/V411_ONE_UPDATE_ACCEPTED"
printf 'V411_V169_V409_ONE_UPDATE_ACCEPTED\n'
