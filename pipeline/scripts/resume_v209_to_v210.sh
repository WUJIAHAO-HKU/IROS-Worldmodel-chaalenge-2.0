#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
P="$BASE/pipeline/scripts"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'
NAME='v210_v209_conservativekl_h200_r2_step8_lr5e6_beta005_seed1408_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
FROZEN="$OFF/frozen_candidates/v210_step8_seed1408"
FREEZE="$FROZEN/freeze_manifest.json"
SEEDS="$OFF/real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
FINAL_OUTPUT="$OFF/real_robotwin_eval/frozen_v210_step8_seed1408_final128"

test -s "$OFF/run_registry/v209_v202_v208_public_arm_routed_release/manifest_compatibility_correction.json"
bash "$P/restart_v209_services.sh" start
"$GO1" "$P/prepare_v210_rl.py"
echo "V210_START $(date -Is)"
bash "$P/launch_v210_pipeline.sh"
test -f "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
test ! -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
bash "$P/freeze_v210_candidate.sh"
"$PY" "$P/prepare_frozen_final128_prereg.py" \
  --freeze-manifest "$FREEZE" --final-seed-manifest "$SEEDS" \
  --output-root "$FINAL_OUTPUT" --output "$FROZEN/final128_preregistration.json"
echo "V210_FINAL128_START $(date -Is)"
set +e
bash "$P/run_frozen_v210_final128_once.sh"
rc=$?
set -e
if (( rc == 0 )); then echo "V210_FINAL128_TARGET_REACHED $(date -Is)"; elif (( rc == 2 )); then echo "V210_FINAL128_TARGET_NOT_REACHED $(date -Is)"; else exit "$rc"; fi
exit "$rc"
