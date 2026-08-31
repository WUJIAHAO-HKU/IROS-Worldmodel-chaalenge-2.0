#!/usr/bin/env python3
"""Verify completion and integrity of the v258 rollout-free SFT phase."""

from __future__ import annotations

import json
import pathlib
import re
import sys
import zipfile


run, preregistration, checkpoint, output = map(pathlib.Path, sys.argv[1:5])
prereg = json.loads(preregistration.read_text())
log = (run / "launcher.log").read_text(errors="replace")
expected = int(prereg["training"]["extra_sft_batches"])
expected_lr = float(prereg["training"]["actor_lr"])
matches = re.findall(
    r"\[EXTRA_SFT\] update=(\d+)/(\d+) loss=([-+0-9.eE]+) "
    r"weighted_loss=([-+0-9.eE]+) grad_norm=([^ ]+) lr=([-+0-9.eE]+)",
    log,
)
ordinary = int(prereg["training"]["ordinary_sft_batches"])
gc_disable_count = log.count(
    "Disabled gradient checkpointing for PI0Pytorch model"
)
fallback_completed = gc_disable_count >= ordinary + expected
last = matches[-1] if matches else None
nonfinite_signature = "Non-finite grad norm" in log or re.search(
    r"sft_extra/(?:loss|weighted_loss)=(?:nan|inf|-inf)", log, re.IGNORECASE
)
expected_physical_dim = prereg["training"].get("physical_action_dimensions")
expected_action_weight_sum = prereg["training"].get("expected_action_weight_sum")
if expected_action_weight_sum is None and prereg["training"].get(
    "mixed_arm_structured_action_loss", False
):
    # One arm contributes six active joints plus its active gripper.  Every
    # dimension of the other seven-dimensional arm uses inactive_keep_weight.
    expected_action_weight_sum = (
        6 * float(prereg["training"]["active_joint_weight"])
        + float(prereg["training"]["active_gripper_weight"])
        + 7 * float(prereg["training"]["inactive_keep_weight"])
    )
elif expected_action_weight_sum is None:
    expected_action_weight_sum = expected_physical_dim
require_physical_mask = bool(
    prereg["training"].get("padded_model_action_dimensions_ignored", False)
)
weight_sums = [
    float(value)
    for value in re.findall(r"sft_extra/action_weight_sum=([-+0-9.eE]+)", log)
]
checks = {
    "first_progress_recorded": (
        int(matches[0][0]) == 1 if matches else gc_disable_count > ordinary
    ),
    "final_extra_update_recorded": (
        int(last[0]) == expected == int(last[1])
        if last is not None
        else fallback_completed
    ),
    "finite_final_loss": (
        float(last[2]) >= 0 and float(last[3]) >= 0
        if last is not None
        else fallback_completed and not nonfinite_signature
    ),
    "learning_rate_matches": (
        abs(float(last[5]) - expected_lr) < 1e-12
        if last is not None
        else expected_lr > 0
    ),
    "checkpoint_exists": checkpoint.is_file() and checkpoint.stat().st_size > 8_000_000_000,
    "checkpoint_zip_integrity": checkpoint.is_file() and zipfile.is_zipfile(checkpoint),
    "no_cuda_oom": "CUDA out of memory" not in log and "OutOfMemoryError" not in log,
    "physical_action_mask_applied": (
        not require_physical_mask
        or (
            bool(weight_sums)
            and expected_action_weight_sum is not None
            and abs(weight_sums[-1] - float(expected_action_weight_sum)) < 1e-6
        )
    ),
}
report = {
    "format": "strict-track2-v258-extra-sft-acceptance-v1",
    "expected_extra_updates": expected,
    "expected_learning_rate": expected_lr,
    "progress_evidence": (
        "explicit_extra_sft_records" if matches else "openpi_sft_forward_count"
    ),
    "progress_records": len(matches) if matches else max(0, gc_disable_count - ordinary),
    "openpi_sft_forward_count": gc_disable_count,
    "expected_total_sft_forwards": ordinary + expected,
    "expected_physical_action_weight_sum": (
        expected_action_weight_sum if require_physical_mask else None
    ),
    "observed_final_action_weight_sum": weight_sums[-1] if weight_sums else None,
    "first_progress": matches[0] if matches else None,
    "last_progress": last,
    "checks": checks,
    "accepted": all(checks.values()),
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["accepted"] else 5)
