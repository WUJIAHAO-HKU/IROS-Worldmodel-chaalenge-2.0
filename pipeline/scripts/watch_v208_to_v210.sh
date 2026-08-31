#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'
V208="$JOINT/v208_v205_mixed_right_gripper_contrast_long32_seed1407"
V210_NAME='v210_v209_conservativekl_h200_r2_step8_lr5e6_beta005_seed1408_20260818'
V210_REG="$OFF/run_registry/$V210_NAME"
V210_RUN="$OFF/runs/$V210_NAME"
FROZEN="$OFF/frozen_candidates/v210_step8_seed1408"
FREEZE="$FROZEN/freeze_manifest.json"
SEEDS="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_OUTPUT="$OFF/real_robotwin_eval/frozen_v210_step8_seed1408_final128"

echo "WAIT_V208 $(date -Is)"
while [[ ! -f "$V208/V208_RIGHT_PARENT_ACCEPTED" && ! -f "$V208/V208_RIGHT_PARENT_REJECTED" ]]; do
  if ! screen -ls 2>/dev/null | grep -q '[.]v208_resume'; then
    echo "V208_PIPELINE_EXITED_WITHOUT_DECISION $(date -Is)" >&2
    exit 4
  fi
  sleep 15
done
if [[ -f "$V208/V208_RIGHT_PARENT_REJECTED" ]]; then
  echo "V208_REJECTED_STOP $(date -Is)"
  exit 2
fi

while screen -ls 2>/dev/null | grep -q '[.]v208_resume'; do sleep 2; done
echo "V208_ACCEPTED $(date -Is)"
"$GO1" "$P/package_v209_arm_routed_release.py"
bash "$P/restart_v209_services.sh" start
"$GO1" "$P/prepare_v210_rl.py"
echo "V210_START $(date -Is)"
bash "$P/launch_v210_pipeline.sh"

test -f "$V210_RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
test ! -e "$V210_RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
bash "$P/freeze_v210_candidate.sh"
test -s "$FREEZE"
"$PY" "$P/prepare_frozen_final128_prereg.py" \
  --freeze-manifest "$FREEZE" --final-seed-manifest "$SEEDS" \
  --output-root "$FINAL_OUTPUT" --output "$FROZEN/final128_preregistration.json"
echo "V210_FINAL128_START $(date -Is)"
set +e
bash "$P/run_frozen_v210_final128_once.sh"
final_rc=$?
set -e
if (( final_rc == 0 )); then
  echo "V210_FINAL128_TARGET_REACHED $(date -Is)"
elif (( final_rc == 2 )); then
  echo "V210_FINAL128_TARGET_NOT_REACHED $(date -Is)"
else
  exit "$final_rc"
fi
exit "$final_rc"
