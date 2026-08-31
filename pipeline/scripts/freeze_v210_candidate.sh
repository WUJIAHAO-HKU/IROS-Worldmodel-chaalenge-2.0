#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v210_v209_conservativekl_h200_r2_step8_lr5e6_beta005_seed1408_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
PARENT_MANIFEST="$JOINT/v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json"
FINAL_RUNNER="$BASE/pipeline/scripts/run_strict_track2_final128_eval.sh"
FINAL_SUMMARIZER="$BASE/pipeline/scripts/summarize_frozen_final128.py"
FINAL_PREREG_TOOL="$BASE/pipeline/scripts/prepare_frozen_final128_prereg.py"
FINAL_WRAPPER="$BASE/pipeline/scripts/run_frozen_v210_final128_once.sh"
FREEZE="$OFF/frozen_candidates/v210_step8_seed1408/freeze_manifest.json"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

test -f "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"
test -f "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
test ! -e "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"
test ! -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
for path in "$CHECKPOINT" "$REG/preregistration.json" \
  "$RUN/audit/p3_training_acceptance.json" \
  "$REG/public_policy_screen_preregistration.json" \
  "$RUN/audit/public_policy_stage_a_acceptance.json" \
  "$RUN/audit/public_policy_stage_b_acceptance.json" "$PARENT_MANIFEST" \
  "$FINAL_RUNNER" "$FINAL_SUMMARIZER" "$FINAL_PREREG_TOOL" "$FINAL_WRAPPER"; do
  test -s "$path"
done
test ! -e "$FREEZE"
if find "$OFF/frozen_candidates" -type f -name freeze_manifest.json -print -quit \
    2>/dev/null | grep -q .; then
  echo 'REFUSING_FREEZE: another unique final candidate already exists' >&2
  exit 3
fi

mkdir -p "$(dirname "$FREEZE")"
"$PY" - "$CHECKPOINT" "$REG/preregistration.json" \
  "$RUN/audit/p3_training_acceptance.json" \
  "$REG/public_policy_screen_preregistration.json" \
  "$RUN/audit/public_policy_stage_a_acceptance.json" \
  "$RUN/audit/public_policy_stage_b_acceptance.json" "$PARENT_MANIFEST" \
  "$FINAL_RUNNER" "$FINAL_SUMMARIZER" "$FINAL_PREREG_TOOL" \
  "$FINAL_WRAPPER" "$FREEZE" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

paths = list(map(Path, sys.argv[1:]))
(checkpoint, training_prereg, training_audit, screen_prereg, stage_a_path,
 stage_b_path, parent, runner, summarizer, prereg_tool, wrapper, output) = paths

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

training = json.loads(training_audit.read_text())
screen = json.loads(screen_prereg.read_text())
stage_a = json.loads(stage_a_path.read_text())
stage_b = json.loads(stage_b_path.read_text())
assert training.get("passed") is True
assert stage_a.get("passed") is True and stage_a.get("stage") == "stage_a"
assert stage_b.get("passed") is True and stage_b.get("stage") == "stage_b"
assert screen["candidate"]["variant"] == "v210_step8_seed1408"
assert stage_b["candidate"]["all"]["count"] == 112
files = {
    "training_preregistration": training_prereg,
    "training_audit": training_audit,
    "public_screen_preregistration": screen_prereg,
    "public_stage_a_audit": stage_a_path,
    "public_stage_b_audit": stage_b_path,
    "local_final128_runner": runner,
    "local_final128_summarizer": summarizer,
    "local_final128_preregistration_tool": prereg_tool,
    "local_final128_one_candidate_wrapper": wrapper,
}
payload = {
    "format": "strict-track2-unique-final-candidate-freeze-v2",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "variant": "v210_step8_seed1408",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest(checkpoint),
    "parent_world_model_manifest": str(parent),
    "parent_world_model_manifest_sha256": digest(parent),
    "evidence": {name: {"path": str(path), "sha256": digest(path)} for name, path in files.items()},
    "public_stage_a_successes": stage_a["candidate"]["all"]["successes"],
    "public_stage_b_successes": stage_b["candidate"]["all"]["successes"],
    "unique_final_candidate": True,
    "official_submission": False,
    "local_final128_runs_for_this_selection_protocol_before_freeze": 0,
    "selection_inputs": "training diagnostics and public train-seed simulator outcomes only",
    "post_final_selection_allowed": False,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED_AND_FROZEN"
echo "V210_UNIQUE_CANDIDATE_FROZEN manifest=$FREEZE"
