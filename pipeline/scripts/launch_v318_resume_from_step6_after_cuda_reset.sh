#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
JOINT="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
EXP='wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05'
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'
RELEASE="$JOINT/v317_v315_native_batch_causal_gate_seed1490_20260822"
MANIFEST="$RELEASE/release_registration.json"
AUTH="$RELEASE/audit/expensive_rl_authorization.json"
REFERENCE="$OFF/immutable_reference_policy_state/pi05_official_rank0.pt"
GATE="$RUN/audit/v318_global_step6_continuation_gate.json"
CKPT="$RUN/$EXP/checkpoints/global_step_6/actor/model_state_dict/full_weights.pt"
OPTIMIZER="$RUN/$EXP/checkpoints/global_step_6/actor/optimizer_recovery.pt"
TERMINAL="$RUN/$EXP/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt"
FAILED_LOG="$REG/step3_resume_launcher_attempt_1.log"
SERVICE_LOG="$REG/v317_service.log"
RECOVERY="$REG/global_step6_cuda_reset_recovery_preregistration.json"

for path in "$REG/preregistration.json" "$AUTH" "$MANIFEST" "$REFERENCE" "$GATE" "$CKPT" "$OPTIMIZER" "$FAILED_LOG" "$SERVICE_LOG"; do
  test -s "$path"
done

# A poisoned CUDA context is a hard stop.  Do not enter the retry loop or
# silently substitute the older v271 service.
for python in "$GO1" "$PY"; do
  CUDA_VISIBLE_DEVICES=0 "$python" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.arange(1024, device="cuda", dtype=torch.float32)
assert x.sum().item() == 523776.0
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
PY
done

if [[ ! -s "$RECOVERY" ]]; then
  "$PY" "$P/prepare_v318_step6_cuda_recovery.py" \
    --run "$RUN" --registry "$REG" --preregistration "$REG/preregistration.json" \
    --step6-gate "$GATE" --failed-log "$FAILED_LOG" --checkpoint "$CKPT" \
    --service-log "$SERVICE_LOG" --optimizer "$OPTIMIZER" --output "$RECOVERY" \
    >"$REG/global_step6_cuda_reset_recovery_prepare.log" 2>&1
fi

"$PY" - "$GATE" "$AUTH" "$RECOVERY" "$CKPT" "$OPTIMIZER" <<'PY'
import json,sys,zipfile
gate=json.load(open(sys.argv[1])); auth=json.load(open(sys.argv[2])); recovery=json.load(open(sys.argv[3]))
assert gate.get("passed") is True and all(gate.get("checks", {}).values())
assert auth.get("passed") is True and auth.get("launch_permission") is True
assert recovery.get("format") == "strict-track2-v318-step6-cuda-reset-recovery-v1"
invariants=recovery.get("invariants", {})
for key in ("algorithm_changed", "world_model_changed", "public112_or_final_outcomes_used", "real_submission"):
    assert invariants.get(key) is False
for key in ("same_run_directory", "same_model_and_optimizer_state"):
    assert invariants.get(key) is True
assert zipfile.is_zipfile(sys.argv[4]) and zipfile.is_zipfile(sys.argv[5])
PY

restore() {
  "$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
  bash "$P/restart_v317_services.sh" stop >/dev/null 2>&1 || true
}
trap restore EXIT

TRACK2_SERVICE_REG="$REG" TRACK2_CPUSET=0-5 TRACK2_WAM_RELEASE_CUDA_CACHE=1 \
  bash "$P/restart_v317_services.sh" start >"$REG/restart_v317_after_cuda_reset.log" 2>&1
curl -fsS http://127.0.0.1:18084/health | grep -q '"status":"ready"'
curl -fsS http://127.0.0.1:8005/v1/health | grep -q 'track2-v317-batched-sparse-failure-terminal-v315'

export TRACK2_ROOT="$ROOT" TRACK2_RUN="$RUN"
export TRACK2_MODEL_VERSION='track2-v317-batched-sparse-failure-terminal-v315'
export TRACK2_BRIDGE_URL='http://127.0.0.1:18084' TRACK2_WAM_URL='http://127.0.0.1:8005'
export TRACK2_PARENT_MODEL="$MANIFEST"
export TRACK2_MAX_STEPS=10 TRACK2_SAVE_INTERVAL=3 TRACK2_KEEP_LAST_CHECKPOINTS=2
export TRACK2_GROUP_SIZE=4 TRACK2_TOTAL_ENVS=32 TRACK2_ACTOR_SEED=1471 TRACK2_ENV_SEED=0
export TRACK2_ACTOR_LR=2e-5 TRACK2_KL_BETA=.01 TRACK2_KL_PENALTY=low_var_kl
export TRACK2_MAX_EPISODE_STEPS=200 TRACK2_MAX_STEPS_PER_ROLLOUT_EPOCH=200 TRACK2_ROLLOUT_EPOCH=4
export TRACK2_ACTOR_GLOBAL_BATCH_SIZE=3200
export TRACK2_REFERENCE_STATE_STORAGE=disk TRACK2_REFERENCE_STATE_PATH_OVERRIDE="$REFERENCE"
export TRACK2_RELEASE_REFERENCE_BEFORE_TERMINAL_CHECKPOINT=true TRACK2_TERMINAL_MODEL_ONLY_CHECKPOINT=true
export TRACK2_CATCH_SYSTEM_FAILURE=0 TRACK2_ACTOR_OFFLOAD=true TRACK2_ENABLE_SFT_CO_TRAIN=false
export TOKENIZERS_PARALLELISM=false TRACK2_CPUSET='6-21' TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TRACK2_RAY_SYSTEM_CONFIG_JSON='{"grpc_client_keepalive_timeout_ms":1800000,"grpc_keepalive_timeout_ms":1800000,"health_check_timeout_ms":1800000,"health_check_failure_threshold":10}'
export TRACK2_ALLOW_EXISTING_RUN=true TRACK2_START_STEP=6 TRACK2_CKPT_PATH="$CKPT"
export TRACK2_OPTIMIZER_RECOVERY_PATH="$OPTIMIZER"

printf 'start_utc=%s start_step=6 recovery=%s\n' "$(date -u +%FT%TZ)" "$RECOVERY" \
  >>"$REG/step6_cuda_reset_resume_attempt.log"
set +e
bash "$P/run_strict_track2_conservative_kl.sh" \
  >"$REG/step6_cuda_reset_resume.screen.log" 2>&1
rc=$?
set -e
cp -f "$RUN/launcher.log" "$REG/step6_cuda_reset_resume_launcher.log" 2>/dev/null || true
printf 'exit_rc=%s end_utc=%s\n' "$rc" "$(date -u +%FT%TZ)" \
  >>"$REG/step6_cuda_reset_resume_attempt.log"
if (( rc != 0 )); then
  echo "V318_STEP6_CUDA_RESET_RESUME_FAILED rc=$rc" >&2
  exit "$rc"
fi

test -s "$TERMINAL"
"$PY" -c 'import sys,zipfile; raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$TERMINAL"
"$PY" "$P/audit_strict_track2_kl_smoke.py" --run "$RUN" \
  --preregistration "$REG/preregistration.json" \
  --output "$RUN/audit/p3_training_acceptance.json" \
  >"$RUN/audit/p3_training_acceptance.log" 2>&1
touch "$RUN/audit/V318_TRAINING_ACCEPTED"
echo V318_STEP6_CUDA_RESET_RESUMED_TRAINING_ACCEPTED
