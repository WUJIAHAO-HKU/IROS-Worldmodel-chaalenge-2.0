#!/usr/bin/env python3
"""Audit a preregistered long-horizon parent using internal visual metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mean_prefix(record: dict, length: int) -> float:
    values = [float(value) for value in record["mae_by_prediction_frame"][:length]]
    if len(values) != length:
        raise ValueError(f"expected {length} per-frame metrics, received {len(values)}")
    return sum(values) / length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    preregistration = json.loads(Path(args.preregistration).read_text())
    history = manifest["validation"]
    by_step = {int(record["step"]): record for record in history}
    baseline = by_step[0]
    best_step = int(manifest["best_checkpoint_step"])
    candidate = by_step[best_step]
    gates = preregistration["acceptance_gates"]

    baseline_first8 = mean_prefix(baseline, 8)
    candidate_first8 = mean_prefix(candidate, 8)
    baseline_right_high = baseline["action_right"]["high_motion_rollout_mae"]
    candidate_right_high = candidate["action_right"]["high_motion_rollout_mae"]
    finite_values = [
        baseline["selection_metric"],
        candidate["selection_metric"],
        baseline["rollout_mae"],
        candidate["rollout_mae"],
        baseline_first8,
        candidate_first8,
        baseline["high_motion_rollout_mae"],
        candidate["high_motion_rollout_mae"],
        baseline_right_high,
        candidate_right_high,
    ]
    checks = {
        "metrics_finite": all(value is not None and math.isfinite(float(value)) for value in finite_values),
        "best_requires_update": best_step > 0,
        "selection_improves": candidate["selection_metric"]
        <= baseline["selection_metric"] * gates["selection_relative_max"],
        "rollout16_improves": candidate["rollout_mae"]
        <= baseline["rollout_mae"] * gates["rollout16_relative_max"],
        "first8_nonregression": candidate_first8
        <= baseline_first8 * gates["first8_relative_max"],
        "high_motion_nonregression": candidate["high_motion_rollout_mae"]
        <= baseline["high_motion_rollout_mae"] * gates["high_motion_relative_max"],
        "right_high_motion_nonregression": candidate_right_high
        <= baseline_right_high * gates["right_high_motion_relative_max"],
    }
    report = {
        "format": "strict-track2-public-long16-parent-audit-v1",
        "selection_inputs": "episode-disjoint public visual metrics and joint14 arm strata only",
        "manifest": str(Path(args.manifest).resolve()),
        "preregistration": str(Path(args.preregistration).resolve()),
        "baseline_step": 0,
        "candidate_step": best_step,
        "baseline": {
            "selection_metric": baseline["selection_metric"],
            "rollout_mae": baseline["rollout_mae"],
            "first8_mean_mae": baseline_first8,
            "frame16_mae": baseline["mae_by_prediction_frame"][15],
            "high_motion_rollout_mae": baseline["high_motion_rollout_mae"],
            "right_high_motion_rollout_mae": baseline_right_high,
        },
        "candidate": {
            "selection_metric": candidate["selection_metric"],
            "rollout_mae": candidate["rollout_mae"],
            "first8_mean_mae": candidate_first8,
            "frame16_mae": candidate["mae_by_prediction_frame"][15],
            "high_motion_rollout_mae": candidate["high_motion_rollout_mae"],
            "right_high_motion_rollout_mae": candidate_right_high,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
