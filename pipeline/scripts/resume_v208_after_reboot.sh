#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
OFF="$BASE/artifacts/strict_track2_official_20260810"
RUN="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
REG="$OFF/run_registry/v208_v205_mixed_right_gripper_contrast_long32_seed1407/preregistration.json"
AMENDMENT="$OFF/run_registry/v208_v205_mixed_right_gripper_contrast_long32_seed1407/zero_step_training_amendment.json"
AMENDMENT_V2="$OFF/run_registry/v208_v205_mixed_right_gripper_contrast_long32_seed1407/zero_step_training_amendment_v2.json"
P="$BASE/pipeline/scripts"
RLINF="$BASE/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
PY='/root/miniconda3/envs/go1/bin/python'
LEFT="$JOINT/v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
RIGHT="$JOINT/v205_v202_public_right_terminal_multichunk_seed1405/selected_right_expert"
MIXED="$JOINT/v163_mixed_reward_windows"
ONPOLICY="$JOINT/onpolicy_windows_full128_stride4"
FAILURE_SPLIT="$RUN/audit/failure_holdout_split.json"
INSTRUCTIONS="$JOINT/reward_alignment/exact_instruction_map128.json"
REWARD="$BASE/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$BASE/artifacts/official_resources/reward_model/t5-base"
RESET="$BASE/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
export PYTHONPATH="$BASE/pipeline:$BASE/pipeline/scripts:$RLINF"

cleanup() {
  /root/autodl-tmp/conda_envs/rlinf_track2/bin/ray stop --force >/dev/null 2>&1 || true
  cd "$BASE"
  bash "$P/restart_v206_services.sh" >"$RUN/restart_v206_after_v208.log" 2>&1 || true
}
trap cleanup EXIT

test -s "$REG"
test "$(sha256sum "$REG" | awk '{print $1}')" = '48e22442f55c5fb8c1d50a4c41c9cdb6ea5410212915915fc3a7e7c06e3b6016'
test -s "$AMENDMENT"
test -s "$AMENDMENT_V2"
test -s "$RUN/audit/baseline/public_success_holdout.npz"
test -s "$RUN/audit/baseline/public_success_reward.json"
"$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$RUN/audit/baseline/public_success_reward.json"
if [[ ! -s "$FAILURE_SPLIT" ]]; then
  "$PY" "$P/prepare_v208_failure_holdout.py"
fi
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["failure_episode_count"] == 10; assert x["right_failure_episode_count"] == 7; assert x["episode_disjoint_from_training"] is True' "$FAILURE_SPLIT"

screen -S wm_v206_bridge -X quit 2>/dev/null || true
screen -S wm_v206_gpu -X quit 2>/dev/null || true
/root/autodl-tmp/conda_envs/rlinf_track2/bin/ray stop --force >/dev/null 2>&1 || true

if [[ ! -s "$RUN/audit/baseline/public_failure_holdout.npz" ]]; then
  "$PY" "$P/export_strict_track2_p2_reward_cache.py" \
    --windows "$ONPOLICY" --split-manifest "$FAILURE_SPLIT" \
    --instruction-map "$INSTRUCTIONS" \
    --baseline-left "$LEFT" --baseline-right "$RIGHT" \
    --candidate-left "$LEFT" --candidate-right "$RIGHT" \
    --output "$RUN/audit/baseline/public_failure_holdout.npz" \
    --chunks 4 --max-sequences 64 --device cuda \
    >"$RUN/audit/baseline/public_failure_export.log" 2>&1
fi
if [[ ! -s "$RUN/audit/baseline/public_failure_reward.json" ]]; then
  "$PY" "$P/evaluate_strict_track2_p2_reward_alignment.py" \
    --cache "$RUN/audit/baseline/public_failure_holdout.npz" \
    --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
    --output "$RUN/audit/baseline/public_failure_reward.json" --batch-size 32 --device cuda \
    >"$RUN/audit/baseline/public_failure_reward.log" 2>&1
fi
"$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$RUN/audit/baseline/public_failure_reward.json"
touch "$RUN/audit/baseline/COMPLETE"
echo V208_BASELINE_RESUME_COMPLETE

if [[ ! -f "$RUN/TRAINING_COMPLETE" ]]; then
  test ! -e "$RUN/checkpoints"
  bash "$P/run_v208_training.sh"
fi
if [[ ! -f "$RUN/CANDIDATE_AUDIT_COMPLETE" ]]; then
  bash "$P/run_v208_candidate_audit.sh"
fi
if [[ ! -f "$RUN/V208_RIGHT_PARENT_ACCEPTED" && ! -f "$RUN/V208_RIGHT_PARENT_REJECTED" ]]; then
  "$PY" "$P/select_v208_mixed_right_contrast.py"
fi
echo V208_PIPELINE_RESUME_COMPLETE
