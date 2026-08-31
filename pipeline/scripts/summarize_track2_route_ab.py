#!/usr/bin/env python3
"""Summarize frozen-V16.9 right-arm routing A/B on the 8-seed shard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRIC = re.compile(r"'eval/(?P<key>[a-zA-Z0-9_]+)':\s*(?:array\()?\s*(?P<value>[0-9.eE+-]+)")


def parse(path: Path, count: int = 8) -> dict:
    lines = [line for line in path.read_text(errors="replace").splitlines() if "'eval/num_trajectories'" in line]
    if not lines:
        raise RuntimeError(f"incomplete route result: {path}")
    values = {m.group("key"): float(m.group("value")) for m in METRIC.finditer(lines[-1])}
    keys = ("success_once", "success_at_end", "grasp_once", "arm_left", "arm_right",
            "left_success", "right_success", "left_grasp", "right_grasp")
    available = {key: int(round(values[key] * count)) for key in keys if key in values}
    return {"count": count, "log": str(path.resolve()),
            "counts": available,
            "rates": {key: values[key] for key in keys if key in values}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prior = json.loads(args.reference_summary.read_text())
    ref_name = prior["candidate_variant"]
    ref = next(row for row in prior["primary_official_success"][ref_name]["batches"] if row["batch"] == "00")
    results = {}
    for variant in args.variants:
        results[variant] = parse(args.metrics_root / variant / "batch16_00" / "launcher.log")
        results[variant]["improvement_vs_frozen_v169"] = {
            "success_count": results[variant]["counts"].get("success_once", -1) - ref["successes"],
            "success_percentage_points": 100 * (results[variant]["counts"].get("success_once", -1) - ref["successes"]) / ref["count"],
            "arm_metrics_available": all(key in results[variant]["counts"] for key in ("arm_left", "arm_right", "left_success", "right_success")),
        }
    report = {
        "format": "strict-track2-frozen-v169-route-ab-v1",
        "reference": {"variant": ref_name, "count": ref["count"], "successes": ref["successes"]},
        "results": results,
        "selection_rule": "route must improve frozen V16.9 shard success and not reduce left-arm success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
