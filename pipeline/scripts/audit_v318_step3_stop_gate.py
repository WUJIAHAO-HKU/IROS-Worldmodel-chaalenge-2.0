#!/usr/bin/env python3
"""Audit the preregistered v318 global-step-3 continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--prefix-audit", required=True, type=Path)
    parser.add_argument("--offline-audit", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--optimizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    prereg = json.loads(args.preregistration.read_text())
    prefix = json.loads(args.prefix_audit.read_text())
    offline = json.loads(args.offline_audit.read_text())
    thresholds = prereg["thresholds"]

    steps = prefix["common_env_steps"]
    success = prefix["metrics"]["env/success_once"]
    returns = prefix["metrics"]["env/return"]
    first3_success = [float(success[str(step)]) for step in (0, 1, 2)]
    first3_returns = [float(returns[str(step)]) for step in (0, 1, 2)]
    official = offline["checkpoints"]["official"]["metrics"]
    candidate = offline["checkpoints"]["v318_step3"]["metrics"]

    ratios = {
        "sample_mse_over_official": candidate["all"]["sample_mse_first8_physical14"]
        / official["all"]["sample_mse_first8_physical14"],
        "right_active_mse_over_official": candidate["right"]["active_arm_mse"]
        / official["right"]["active_arm_mse"],
        "inactive_mse_over_official": candidate["all"]["inactive_arm_mse"]
        / official["all"]["inactive_arm_mse"],
    }
    guards = prereg["guards"]
    checks = {
        "preregistered_guards_hold": (
            guards["public_train40_actions_only_for_offline_diagnostic"] is True
            and guards["thresholds_registered_before_step2_metrics_or_checkpoint_diagnostic"] is True
            and guards["public112_outcomes_read"] is False
            and guards["reserved_final128_outcomes_read"] is False
            and guards["hidden_outcomes_read"] is False
            and guards["real_submission"] is False
            and guards["algorithm_or_budget_changed"] is False
        ),
        "prefix_audit_passed": prefix.get("passed") is True,
        "exact_metric_steps_0_1_2": steps == [0, 1, 2] and prefix["common_train_steps"] == [0, 1, 2],
        "step2_success_once_min": first3_success[2] >= thresholds["step2_success_once_min"],
        "first3_success_once_sum_min": sum(first3_success) >= thresholds["first3_success_once_sum_min"],
        "first3_return_mean_min": sum(first3_returns) / 3 >= thresholds["first3_return_mean_min"],
        "left_route_accuracy_min": candidate["left"]["route_accuracy"] >= thresholds["left_route_accuracy_min"],
        "right_route_accuracy_min": candidate["right"]["route_accuracy"] >= thresholds["right_route_accuracy_min"],
        "sample_mse_ratio_max": ratios["sample_mse_over_official"] <= thresholds["candidate_sample_mse_over_official_max"],
        "right_active_mse_ratio_max": ratios["right_active_mse_over_official"] <= thresholds["candidate_right_active_mse_over_official_max"],
        "inactive_mse_ratio_max": ratios["inactive_mse_over_official"] <= thresholds["candidate_inactive_mse_over_official_max"],
        "all_decision_metrics_finite": all(
            math.isfinite(value)
            for value in first3_success + first3_returns + list(ratios.values())
        ),
        "checkpoint_zip_valid": args.checkpoint.is_file() and zipfile.is_zipfile(args.checkpoint),
        "optimizer_zip_valid": args.optimizer.is_file() and zipfile.is_zipfile(args.optimizer),
    }
    report = {
        "format": "strict-track2-v318-global-step3-continuation-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": {"path": str(args.preregistration), "sha256": sha256(args.preregistration)},
        "prefix_audit": {"path": str(args.prefix_audit), "sha256": sha256(args.prefix_audit)},
        "offline_audit": {"path": str(args.offline_audit), "sha256": sha256(args.offline_audit)},
        "checkpoint": {"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size, "sha256": sha256(args.checkpoint)},
        "optimizer": {"path": str(args.optimizer), "bytes": args.optimizer.stat().st_size, "sha256": sha256(args.optimizer)},
        "first3_success_once": first3_success,
        "first3_success_count_equivalent": [round(value * 128) for value in first3_success],
        "first3_return": first3_returns,
        "ratios": ratios,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": "training metrics and public train40 actions only; no public112/final128/hidden/contest outcomes",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
