#!/usr/bin/env python3
"""Apply preregistered paired gates to a parent world-model candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value(report: dict, group: str, metric: str) -> float:
    return float(report["improvement"][group][f"{metric}_improvement_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onpolicy-report", required=True)
    parser.add_argument("--demo-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    onpolicy_path, demo_path = Path(args.onpolicy_report), Path(args.demo_report)
    onpolicy, demo = json.loads(onpolicy_path.read_text()), json.loads(demo_path.read_text())
    artifact_key = next(
        (key for key in ("candidate_checkpoint", "gated_release", "candidate_release") if onpolicy.get(key)),
        None,
    )
    if artifact_key is None:
        raise ValueError("on-policy report does not identify its candidate artifact")
    checks = [
        ("onpolicy.overall.rgb", value(onpolicy, "overall", "rgb_mae"), 3.0),
        ("onpolicy.right.contact", value(onpolicy, "right", "contact_rgb_mae"), 3.0),
        ("onpolicy.left.rgb", value(onpolicy, "left", "rgb_mae"), -0.5),
        ("onpolicy.overall.texture", value(onpolicy, "overall", "texture_mae"), -0.5),
        ("onpolicy.overall.temporal", value(onpolicy, "overall", "temporal_delta_mae"), -0.5),
    ]
    for group in ("overall", "left", "right"):
        checks.extend([
            (f"demo.{group}.rgb", value(demo, group, "rgb_mae"), -0.5),
            (f"demo.{group}.contact", value(demo, group, "contact_rgb_mae"), -0.5),
        ])
    checks.extend([
        ("demo.overall.texture", value(demo, "overall", "texture_mae"), -0.5),
        ("demo.overall.temporal", value(demo, "overall", "temporal_delta_mae"), -0.5),
    ])
    records = [
        {"name": name, "value_percent": measured, "minimum_percent": minimum,
         "passed": measured >= minimum}
        for name, measured, minimum in checks
    ]
    report = {
        "format": "strict-track2-parent-candidate-screen-v1",
        "passed": all(record["passed"] for record in records),
        "checks": records,
        "failed_checks": [record["name"] for record in records if not record["passed"]],
        "onpolicy_report": str(onpolicy_path.resolve()),
        "onpolicy_report_sha256": sha256(onpolicy_path),
        "demo_report": str(demo_path.resolve()),
        "demo_report_sha256": sha256(demo_path),
        "candidate_artifact": {
            "kind": artifact_key,
            "path": onpolicy[artifact_key],
        },
    }
    if artifact_key == "candidate_checkpoint":
        report["candidate_checkpoint"] = onpolicy[artifact_key]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
