#!/usr/bin/env python3
"""Apply the frozen stage-A or stage-B policy-screen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("stage_a", "stage_b"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    stage = prereg[args.stage]
    gate = stage["gate"]
    expected_count = int(stage["count"])
    if int(summary["seed_count"]) != expected_count:
        raise RuntimeError("summary seed count does not match preregistration")
    candidates = summary["candidates"]
    if len(candidates) != 1 or candidates[0]["variant"] != prereg["candidate"]["variant"]:
        raise RuntimeError("summary candidate does not match preregistration")

    baseline = summary["baseline"]
    candidate = candidates[0]["metrics"]
    checks = {
        "candidate_successes_min": (
            candidate["all"]["successes"] >= int(gate["candidate_successes_min"])
        ),
        "success_gain_over_baseline_min": (
            candidate["all"]["successes"] - baseline["all"]["successes"]
            >= int(gate["success_gain_over_baseline_min"])
        ),
        "left_success_rate_min": (
            candidate["left"]["success_rate"] >= float(gate["left_success_rate_min"])
        ),
        "right_success_rate_min": (
            candidate["right"]["success_rate"] >= float(gate["right_success_rate_min"])
        ),
    }
    if gate.get("grasp_completions_not_below_baseline"):
        checks["grasp_completions_not_below_baseline"] = (
            candidate["all"]["grasp_completions"]
            >= baseline["all"]["grasp_completions"]
        )
    if gate.get("grasp_completions_at_least_successes"):
        checks["grasp_completions_at_least_successes"] = (
            candidate["all"]["grasp_completions"]
            >= candidate["all"]["successes"]
        )

    report = {
        "format": "strict-track2-v203b-public-policy-screen-audit-v1",
        "stage": args.stage,
        "preregistration": str(args.preregistration),
        "summary": str(args.summary),
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": "public local RoboTwin outcomes only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
