#!/usr/bin/env python3
"""Select a dual-arm on-policy AR expert from preregistered validation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def metric(report: dict, group: str, name: str) -> float:
    return float(report["improvement"][group][f"{name}_improvement_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    for value in args.reports:
        path = Path(value).resolve()
        report = json.loads(path.read_text())
        checks = {
            "overall_rgb": metric(report, "overall", "rgb_mae"),
            "left_rgb": metric(report, "left", "rgb_mae"),
            "right_rgb": metric(report, "right", "rgb_mae"),
            "overall_contact": metric(report, "overall", "contact_rgb_mae"),
            "left_contact": metric(report, "left", "contact_rgb_mae"),
            "right_contact": metric(report, "right", "contact_rgb_mae"),
            "overall_texture": metric(report, "overall", "texture_mae"),
            "overall_temporal": metric(report, "overall", "temporal_delta_mae"),
        }
        passed = (
            min(checks["overall_rgb"], checks["left_rgb"], checks["right_rgb"]) >= 3
            and min(checks["overall_contact"], checks["left_contact"], checks["right_contact"]) >= 3
            and checks["overall_texture"] >= 0
            and checks["overall_temporal"] >= 0
        )
        worst_arm_task_metric = min(
            checks["left_rgb"], checks["right_rgb"], checks["left_contact"], checks["right_contact"]
        )
        candidates.append({
            "report": str(path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": report["candidate_checkpoint"],
            "passed": passed,
            "metrics": checks,
            "selection_score": worst_arm_task_metric + .01 * checks["overall_rgb"],
        })
    eligible = [item for item in candidates if item["passed"]]
    if not eligible:
        raise RuntimeError("no on-policy expert passed the dual-arm validation gate")
    selected = max(eligible, key=lambda item: item["selection_score"])
    result = {
        "format": "strict-track2-onpolicy-expert-selection-v1",
        "selection_rule": "maximize worst(left/right RGB/contact) + 0.01*overall RGB after all >=3% and texture/temporal nonregression",
        "selected": selected,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
