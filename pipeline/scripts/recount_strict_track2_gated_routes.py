#!/usr/bin/env python3
"""Recompute routing totals when a runtime adds a more specific route suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    report = json.loads(source.read_text())
    records = report["routing"]["records"]
    report["routing"].update(
        {
            "candidate": sum(str(row["route"]).startswith("candidate") for row in records),
            "candidate_left": sum(str(row["route"]).startswith("candidate_left") for row in records),
            "candidate_right": sum(str(row["route"]).startswith("candidate_right") for row in records),
            "baseline": sum(row["route"] == "baseline" for row in records),
            "bit_exact_to_baseline": sum(bool(row["bit_exact_to_baseline"]) for row in records),
        }
    )
    report["routing_recount"] = {
        "source_report": str(source),
        "source_report_sha256": sha256(source),
        "reason": "candidate_blend is a candidate route and was not counted by the older exact-equality summary",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["routing"], indent=2))


if __name__ == "__main__":
    main()
