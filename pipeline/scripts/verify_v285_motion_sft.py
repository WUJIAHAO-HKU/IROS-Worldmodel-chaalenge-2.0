#!/usr/bin/env python3
"""Verify v285 training completion, checkpoint integrity, and motion weighting."""

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
progress = re.findall(
    r"\[EXTRA_SFT\] update=(\d+)/(\d+) loss=([-+0-9.eE]+) "
    r"weighted_loss=([-+0-9.eE]+) grad_norm=([^ ]+) lr=([-+0-9.eE]+)",
    log,
)
action_weight_sums = [
    float(value)
    for value in re.findall(r"sft_extra/action_weight_sum=([-+0-9.eE]+)", log)
]
structured = prereg["training"]["structured_action_weights"]
base_action_weight_sum = (
    6.0 * float(structured["right_joint"])
    + float(structured["right_gripper"])
    + 7.0 * float(structured["inactive_left"])
)
inferred_motion_means = [
    value / base_action_weight_sum for value in action_weight_sums
]
last = progress[-1] if progress else None
checks = {
    "first_progress_recorded": bool(progress) and int(progress[0][0]) == 1,
    "final_extra_update_recorded": bool(last)
    and int(last[0]) == expected == int(last[1]),
    "finite_final_loss": bool(last)
    and float(last[2]) >= 0
    and float(last[3]) >= 0,
    "learning_rate_matches": bool(last)
    and abs(float(last[5]) - expected_lr) < 1e-12,
    # Rich truncates the long direct metric label with a Unicode ellipsis in
    # the final table.  action_weight_sum is untruncated and equals the base
    # structured weight mass times the mean per-example motion scale.
    "motion_weight_metrics_present": bool(action_weight_sums),
    "motion_weight_metrics_bounded": bool(inferred_motion_means)
    and all(
        0.05 - 1e-3 <= value <= 1.0 + 1e-3
        for value in inferred_motion_means
    ),
    "checkpoint_exists": checkpoint.is_file()
    and checkpoint.stat().st_size > 8_000_000_000,
    "checkpoint_zip_integrity": checkpoint.is_file() and zipfile.is_zipfile(checkpoint),
    "no_cuda_oom": "CUDA out of memory" not in log
    and "OutOfMemoryError" not in log,
    "no_nonfinite_gradient": "Non-finite grad norm" not in log,
}
report = {
    "format": "strict-track2-v285-motion-sft-acceptance-v1",
    "expected_extra_updates": expected,
    "progress_records": len(progress),
    "last_progress": last,
    "motion_metric_evidence": "untruncated action_weight_sum / preregistered base weight mass",
    "base_action_weight_sum": base_action_weight_sum,
    "action_weight_sum_records": len(action_weight_sums),
    "inferred_motion_weight_mean_min": (
        min(inferred_motion_means) if inferred_motion_means else None
    ),
    "inferred_motion_weight_mean_max": (
        max(inferred_motion_means) if inferred_motion_means else None
    ),
    "checks": checks,
    "accepted": all(checks.values()),
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["accepted"] else 5)
