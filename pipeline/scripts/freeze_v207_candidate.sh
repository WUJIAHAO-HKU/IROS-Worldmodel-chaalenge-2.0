#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$BASE/artifacts/strict_track2_official_20260810"
JOINT="$BASE/artifacts/strict_track2_joint_augmentation_20260810"
NAME='v207_v206_conservativekl_h200_r2_step8_lr5e6_beta005_seed1406_20260818'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
CHECKPOINT="$RUN/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_8/actor/model_state_dict/full_weights.pt"
TRAINING_PREREG="$REG/preregistration.json"
TRAINING_AUDIT="$RUN/audit/p3_training_acceptance.json"
SCREEN_PREREG="$REG/public_policy_screen_preregistration.json"
STAGE_A_AUDIT="$RUN/audit/public_policy_stage_a_acceptance.json"
STAGE_B_AUDIT="$RUN/audit/public_policy_stage_b_acceptance.json"
PARENT_MANIFEST="$JOINT/v206_v202_v205_public_arm_routed_release/arm_routed_autoregressive_manifest.json"
FINAL_RUNNER="$BASE/pipeline/scripts/run_strict_track2_final128_eval.sh"
FINAL_SUMMARIZER="$BASE/pipeline/scripts/summarize_frozen_final128.py"
FINAL_PREREG_TOOL="$BASE/pipeline/scripts/prepare_frozen_final128_prereg.py"
FINAL_WRAPPER="$BASE/pipeline/scripts/run_frozen_v207_final128_once.sh"
FREEZE="$OFF/frozen_candidates/v207_step8_seed1406/freeze_manifest.json"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'

test -f "$RUN/audit/PUBLIC_POLICY_STAGE_A_PASSED"
test -f "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED"
test ! -e "$RUN/audit/PUBLIC_POLICY_STAGE_A_REJECTED"
test ! -e "$RUN/audit/PUBLIC_POLICY_STAGE_B_REJECTED"
for path in "$CHECKPOINT" "$TRAINING_PREREG" "$TRAINING_AUDIT" \
  "$SCREEN_PREREG" "$STAGE_A_AUDIT" "$STAGE_B_AUDIT" "$PARENT_MANIFEST"; do
  test -s "$path"
done
for path in "$FINAL_RUNNER" "$FINAL_SUMMARIZER" "$FINAL_PREREG_TOOL" \
  "$FINAL_WRAPPER"; do
  test -s "$path"
done
test ! -e "$FREEZE"

# There must not already be a different final candidate.  Once any candidate is
# frozen, final-128 outcomes may never be used to select a replacement.
if find "$OFF/frozen_candidates" -type f -name freeze_manifest.json -print -quit \
    2>/dev/null | grep -q .; then
  echo 'REFUSING_FREEZE: another unique final candidate already exists' >&2
  exit 3
fi

mkdir -p "$(dirname "$FREEZE")"
"$PY" - "$CHECKPOINT" "$TRAINING_PREREG" "$TRAINING_AUDIT" \
  "$SCREEN_PREREG" "$STAGE_A_AUDIT" "$STAGE_B_AUDIT" \
  "$PARENT_MANIFEST" "$FINAL_RUNNER" "$FINAL_SUMMARIZER" \
  "$FINAL_PREREG_TOOL" "$FINAL_WRAPPER" "$FREEZE" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    checkpoint,
    training_prereg,
    training_audit,
    screen_prereg,
    stage_a_audit,
    stage_b_audit,
    parent_manifest,
    final_runner,
    final_summarizer,
    final_prereg_tool,
    final_wrapper,
    output,
) = map(Path, sys.argv[1:])


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


training = json.loads(training_audit.read_text(encoding="utf-8"))
stage_a = json.loads(stage_a_audit.read_text(encoding="utf-8"))
stage_b = json.loads(stage_b_audit.read_text(encoding="utf-8"))
screen = json.loads(screen_prereg.read_text(encoding="utf-8"))
assert training.get("passed") is True, training
assert stage_a.get("passed") is True, stage_a
assert stage_b.get("passed") is True, stage_b
assert stage_a.get("stage") == "stage_a", stage_a
assert stage_b.get("stage") == "stage_b", stage_b
assert screen["candidate"]["variant"] == "v207_step8_seed1406", screen
assert stage_b["candidate"]["all"]["count"] == 112, stage_b

files = {
    "checkpoint": checkpoint,
    "training_preregistration": training_prereg,
    "training_audit": training_audit,
    "public_screen_preregistration": screen_prereg,
    "public_stage_a_audit": stage_a_audit,
    "public_stage_b_audit": stage_b_audit,
    "parent_world_model_manifest": parent_manifest,
    "local_final128_runner": final_runner,
    "local_final128_summarizer": final_summarizer,
    "local_final128_preregistration_tool": final_prereg_tool,
    "local_final128_one_candidate_wrapper": final_wrapper,
}
payload = {
    "format": "strict-track2-unique-final-candidate-freeze-v2",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "variant": "v207_step8_seed1406",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest(checkpoint),
    "parent_world_model_manifest": str(parent_manifest),
    "parent_world_model_manifest_sha256": digest(parent_manifest),
    "evidence": {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in files.items()
        if name not in {"checkpoint", "parent_world_model_manifest"}
    },
    "public_stage_a_successes": stage_a["candidate"]["all"]["successes"],
    "public_stage_b_successes": stage_b["candidate"]["all"]["successes"],
    "unique_final_candidate": True,
    "official_submission": False,
    "local_final128_runs_for_this_selection_protocol_before_freeze": 0,
    "selection_inputs": (
        "training diagnostics and preregistered public train-seed simulator "
        "outcomes only; no hidden or final-128 outcomes"
    ),
    "post_final_selection_allowed": False,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

touch "$RUN/audit/PUBLIC_POLICY_STAGE_B_PASSED_AND_FROZEN"
printf 'V207_UNIQUE_CANDIDATE_FROZEN manifest=%s\n' "$FREEZE"
