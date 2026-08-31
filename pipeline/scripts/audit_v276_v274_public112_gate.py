#!/usr/bin/env python3
"""Audit the preregistered candidate-only 112-seed public gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_strict_track2_dev_eval import aggregate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--batch00-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    first = json.loads(args.batch00_report.read_text(encoding="utf-8"))
    if prereg["variant"] != args.variant or prereg["real_submission"] is not False:
        raise RuntimeError("candidate identity or submission declaration mismatch")
    if prereg["manifest"] != str(args.dev_root / "manifest.json"):
        raise RuntimeError("public112 manifest path mismatch")
    if prereg["manifest_sha256"] != sha256(args.dev_root / "manifest.json"):
        raise RuntimeError("public112 manifest hash mismatch")
    if first.get("accepted") is not True or first.get("reserved_final128_used") is not False:
        raise RuntimeError("batch00 gate did not pass cleanly")

    metrics = aggregate(args.output_root, args.dev_root, args.variant)
    gate = prereg["public112_gate"]
    thresholds = gate["thresholds"]
    all_metrics = metrics["all"]
    batch00 = metrics["batches"][0]
    if all_metrics["count"] != int(gate["count"]):
        raise RuntimeError("public evaluation count mismatch")
    if batch00["batch"] != "00" or batch00["count"] != 16:
        raise RuntimeError("public batch00 identity mismatch")
    if batch00["successes"] != first["counts"]["total_success"]:
        raise RuntimeError("batch00 audit and aggregate disagree")

    checks = {
        "total_success": all_metrics["successes"] >= int(thresholds["total_success"]),
        "left_success_rate": metrics["left"]["success_rate"] >= float(thresholds["left_success_rate"]),
        "right_success_rate": metrics["right"]["success_rate"] >= float(thresholds["right_success_rate"]),
        "grasp_at_least_success": all_metrics["grasp_completions"] >= all_metrics["successes"],
    }
    report = {
        "format": "strict-track2-v276-v274-public112-audit-v1",
        "variant": args.variant,
        "count": all_metrics["count"],
        "candidate": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": "public local RoboTwin train-seed outcomes only",
        "official_submission": False,
        "reserved_final128_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
