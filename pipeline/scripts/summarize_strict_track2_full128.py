#!/usr/bin/env python3
"""Aggregate strict official Track-2 RoboTwin adjust_bottle batch logs.

The evaluator reports rates per batch.  This script converts those rates back
to integer counts using the logged trajectory count, then aggregates the full
128-seed result and arm-stratified metrics without mixing in world-model or
MPC diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC = re.compile(r"'eval/(?P<key>[a-zA-Z0-9_]+)':\s*(?:array\()?\s*(?P<value>[0-9.eE+-]+)")
COUNT = re.compile(r"'eval/num_trajectories':\s*(?P<count>[0-9]+)")
KEYS = (
    "success_once", "success_at_end", "grasp_once", "arm_left", "arm_right",
    "left_success", "right_success", "left_grasp", "right_grasp",
)


def parse_log(path: Path) -> dict:
    lines = [line for line in path.read_text(errors="replace").splitlines()
             if "'eval/num_trajectories'" in line]
    if not lines:
        raise RuntimeError(f"no final metric line in {path}")
    line = lines[-1]
    count_match = COUNT.search(line)
    if count_match is None:
        raise RuntimeError(f"trajectory count missing in {path}")
    metrics = {m.group("key"): float(m.group("value")) for m in METRIC.finditer(line)}
    missing = [key for key in KEYS if key not in metrics]
    if missing:
        raise RuntimeError(f"missing {missing} in {path}")
    counts = {key: int(round(metrics[key] * int(count_match.group("count")))) for key in KEYS}
    return {"count": int(count_match.group("count")), "metrics": metrics,
            "counts": counts, "log": str(path.resolve())}


def aggregate(root: Path, variant: str) -> dict:
    rows = []
    totals = {key: 0 for key in KEYS}
    for batch in range(9):
        expected = 8 if batch in (0, 8) else 16
        path = root / variant / f"batch16_{batch:02d}" / "launcher.log"
        row = parse_log(path)
        if row["count"] != expected:
            raise RuntimeError(f"{path}: expected {expected}, got {row['count']}")
        for key in KEYS:
            totals[key] += row["counts"][key]
        rows.append({"batch": f"{batch:02d}", **row})
    if sum(row["count"] for row in rows) != 128:
        raise RuntimeError("full evaluation does not contain exactly 128 trajectories")
    left_count, right_count = totals["arm_left"], totals["arm_right"]
    return {
        "count": 128,
        "successes": totals["success_once"],
        "success_at_end": totals["success_at_end"],
        "success_rate": totals["success_once"] / 128,
        "grasp_completions": totals["grasp_once"],
        "grasp_completion_rate": totals["grasp_once"] / 128,
        "left": {"count": left_count, "successes": totals["left_success"],
                  "success_rate": totals["left_success"] / left_count if left_count else 0,
                  "grasps": totals["left_grasp"],
                  "grasp_rate": totals["left_grasp"] / left_count if left_count else 0},
        "right": {"count": right_count, "successes": totals["right_success"],
                   "success_rate": totals["right_success"] / right_count if right_count else 0,
                   "grasps": totals["right_grasp"],
                   "grasp_rate": totals["right_grasp"] / right_count if right_count else 0},
        "batches": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--reference-summary", type=Path,
        help="Prior strict summary whose candidate is the frozen reference checkpoint",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_baseline = aggregate(args.eval_root, "baseline")
    candidate = aggregate(args.eval_root, args.candidate)
    reference_label = "baseline"
    reference = raw_baseline
    if args.reference_summary:
        prior = json.loads(args.reference_summary.read_text())
        reference_label = prior["candidate_variant"]
        prior_reference = prior["auxiliary_read_only_metrics"][reference_label]
        reference = {
            "count": prior_reference["count"],
            "successes": prior_reference["successes"],
            "success_rate": prior_reference["success_rate"],
            "grasp_completions": prior_reference["grasp_completions"],
            "grasp_completion_rate": prior_reference["grasp_completion_rate"],
            "left": prior_reference["left"],
            "right": prior_reference["right"],
            "source": str(args.reference_summary.resolve()),
        }
    delta = candidate["successes"] - reference["successes"]
    report = {
        "format": "strict-track2-official-full128-v1",
        "raw_pi05_baseline": raw_baseline,
        "reference_label": reference_label,
        "reference": reference,
        "candidate": candidate,
        "improvement": {
            "success_count": delta,
            "success_percentage_points": 100 * delta / 128,
            "success_relative_percent": 100 * delta / reference["successes"] if reference["successes"] else None,
            "grasp_count": candidate["grasp_completions"] - reference["grasp_completions"],
        },
        "candidate_is_strictly_better": candidate["successes"] > reference["successes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
