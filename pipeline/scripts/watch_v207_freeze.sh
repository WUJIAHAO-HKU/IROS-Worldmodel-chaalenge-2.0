#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v207_v206_conservativekl_h200_r2_step8_lr5e6_beta005_seed1406_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
FREEZER="$BASE/pipeline/scripts/freeze_v207_candidate.sh"
STATUS="$REG/freeze_watch_status.txt"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
FREEZE="$OFF/frozen_candidates/v207_step8_seed1406/freeze_manifest.json"
FINAL_SEED_MANIFEST="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_OUTPUT_ROOT="$OFF/real_robotwin_eval/frozen_v207_step8_seed1406_final128"
FINAL_PREREG="$OFF/frozen_candidates/v207_step8_seed1406/final128_preregistration.json"
FINAL_WRAPPER="$BASE/pipeline/scripts/run_frozen_v207_final128_once.sh"

while screen -ls 2>/dev/null | grep -q '[.]v207_pipeline'; do
  if test -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED" \
      || test -e "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED" \
      || test -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"; then
    break
  fi
  sleep 30
done

if test -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"; then
  # Let the pipeline EXIT trap finish restoring services before the one-shot
  # evaluator takes exclusive ownership of the GPU.
  while screen -ls 2>/dev/null | grep -q '[.]v207_pipeline'; do
    sleep 5
  done
  bash "$FREEZER" > "$REG/freeze_candidate.log" 2>&1
  "$PY" "$BASE/pipeline/scripts/prepare_frozen_final128_prereg.py" \
    --freeze-manifest "$FREEZE" \
    --final-seed-manifest "$FINAL_SEED_MANIFEST" \
    --output-root "$FINAL_OUTPUT_ROOT" \
    --output "$FINAL_PREREG" \
    > "$REG/final128_preregistration.log" 2>&1
  test -x "$FINAL_WRAPPER"
  screen -dmS v207_final128 bash -lc \
    "exec bash \"$FINAL_WRAPPER\" > \"$REG/final128.screen.log\" 2>&1"
  printf 'frozen_preregistered_and_final128_launched\n' > "$STATUS"
elif test -e "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"; then
  printf 'stage_a_rejected\n' > "$STATUS"
elif test -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"; then
  printf 'stage_b_rejected\n' > "$STATUS"
else
  printf 'pipeline_ended_without_public_gate_marker\n' > "$STATUS"
  exit 4
fi
