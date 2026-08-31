#!/usr/bin/env python3
"""Normalize the reused frozen v212 selector output for the v213 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("format") != "strict-track2-v212-long128-logit-audit-v1":
        raise ValueError("unexpected inherited selector report format")
    report["format"] = "strict-track2-v213-long128-logit-audit-v1"
    report["inherited_frozen_selector"] = {
        "path": str(args.selector),
        "sha256": sha256(args.selector),
        "note": "selection logic and gates are unchanged from the completed v212 audit",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    inherited_marker = args.run / (
        "V212_RIGHT_PARENT_ACCEPTED" if report["passed"] else "V212_RIGHT_PARENT_REJECTED"
    )
    if not inherited_marker.exists():
        raise ValueError("inherited selector marker is missing")
    inherited_marker.unlink()
    marker = args.run / (
        "V213_RIGHT_PARENT_ACCEPTED" if report["passed"] else "V213_RIGHT_PARENT_REJECTED"
    )
    marker.touch()
    print(json.dumps({"passed": report["passed"], "marker": str(marker)}, indent=2))


if __name__ == "__main__":
    main()
