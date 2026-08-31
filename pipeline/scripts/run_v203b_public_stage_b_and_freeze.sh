#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
NAME='v203b_v202_conservativekl_h200_r2_step8_seed1403_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
DEV112="$OFF/real_robotwin_eval/public_unseen_train_dev112_seed1403"
OUTPUT="$OFF/real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
VARIANT='v203b_step8_seed1403'
SCREEN_PREREG="$REG/public_policy_screen_preregistration.json"
SUMMARY="$RUN/audit/public_policy_stage_b_summary.json"
AUDIT="$RUN/audit/public_policy_stage_b_acceptance.json"
FREEZE="$OFF/frozen_candidates/v203b_step8_seed1403/freeze_manifest.json"

test -f "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"
test -s "$CHECKPOINT"
test -s "$SCREEN_PREREG"
test ! -e "$FREEZE"

for batch in 02 03 04 05 06; do
  TRACK2_DEV_SEED_ROOT="$DEV112" \
  TRACK2_DEV_OUTPUT_ROOT="$OUTPUT" \
  TRACK2_CANDIDATE_CHECKPOINT="$CHECKPOINT" \
  TRACK2_CANDIDATE_VARIANT="$VARIANT" \
  TRACK2_DEV_BATCH_FILTER="$batch" \
    bash "$BASE/pipeline/scripts/run_strict_track2_dev_eval.sh"
done

"$PY" "$BASE/pipeline/scripts/summarize_strict_track2_dev_eval.py" \
  --output-root "$OUTPUT" \
  --dev-root "$DEV112" \
  --candidate "$VARIANT" \
  --output "$SUMMARY" > "$RUN/audit/public_policy_stage_b_summary.log"

set +e
"$PY" "$BASE/pipeline/scripts/audit_p3_public_policy_screen.py" \
  --preregistration "$SCREEN_PREREG" \
  --summary "$SUMMARY" \
  --stage stage_b \
  --output "$AUDIT" > "$RUN/audit/public_policy_stage_b_acceptance.log" 2>&1
audit_rc=$?
set -e
if (( audit_rc != 0 )); then
  touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
  printf 'V203B_PUBLIC_POLICY_STAGE_B_REJECTED rc=%s\n' "$audit_rc"
  exit "$audit_rc"
fi

mkdir -p "$(dirname "$FREEZE")"
"$PY" - "$CHECKPOINT" "$SCREEN_PREREG" "$AUDIT" "$FREEZE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

checkpoint, prereg, audit, output = map(Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

report = json.loads(audit.read_text(encoding="utf-8"))
assert report["passed"] is True, report
payload = {
    "format": "strict-track2-unique-final-candidate-freeze-v1",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "variant": "v203b_step8_seed1403",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest(checkpoint),
    "public_screen_preregistration": str(prereg),
    "public_screen_preregistration_sha256": digest(prereg),
    "public_stage_b_audit": str(audit),
    "public_stage_b_audit_sha256": digest(audit),
    "unique_final_candidate": True,
    "official_submission": False,
    "reserved_final_128_runs_before_freeze": 0,
    "selection_inputs": "training diagnostics plus public train-seed simulator outcomes only",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED_AND_FROZEN"
printf 'V203B_PUBLIC_POLICY_STAGE_B_PASSED_AND_FROZEN manifest=%s\n' "$FREEZE"
