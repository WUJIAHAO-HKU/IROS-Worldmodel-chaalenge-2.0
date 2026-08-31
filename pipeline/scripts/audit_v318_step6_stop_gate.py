#!/usr/bin/env python3
"""Apply the preregistered v318 global-step-6 continuation gate."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--prefix-audit", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--optimizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    prefix = json.loads(args.prefix_audit.read_text())
    thresholds = prereg["thresholds"]
    success = [float(prefix["metrics"]["env/success_once"][str(step)]) for step in (3, 4, 5)]
    returns = [float(prefix["metrics"]["env/return"][str(step)]) for step in (3, 4, 5)]
    guards = prereg["guards"]
    checks = {
        "guards_hold": (
            guards["registered_before_step4_metrics"] is True
            and guards["training_metrics_only"] is True
            and all(guards[key] is False for key in ("public112_outcomes_read", "final128_outcomes_read", "hidden_outcomes_read", "real_submission", "algorithm_or_budget_changed"))
        ),
        "prefix_passed": prefix.get("passed") is True,
        "exact_steps_0_through_5": prefix["common_env_steps"] == list(range(6)) and prefix["common_train_steps"] == list(range(6)),
        "steps3_5_success_sum_min": sum(success) >= thresholds["steps3_5_success_once_sum_min"],
        "steps3_5_return_mean_min": sum(returns) / 3 >= thresholds["steps3_5_return_mean_min"],
        "step5_success_min": success[2] >= thresholds["step5_success_once_min"],
        "decision_values_finite": all(math.isfinite(value) for value in success + returns),
        "checkpoint_zip_valid": args.checkpoint.is_file() and zipfile.is_zipfile(args.checkpoint),
        "optimizer_zip_valid": args.optimizer.is_file() and zipfile.is_zipfile(args.optimizer),
    }
    report = {
        "format": "strict-track2-v318-global-step6-continuation-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "success_once_steps3_5": success,
        "success_count_equivalent_steps3_5": [round(value * 128) for value in success],
        "return_steps3_5": returns,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": "training-only metrics; no public/final/hidden/contest outcomes",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
