#!/usr/bin/env python3
"""Audit the predeclared two-batch public right-arm screen."""

from __future__ import annotations

import json
import pathlib
import sys


preregistration, summary, output = map(pathlib.Path, sys.argv[1:4])
prereg = json.loads(preregistration.read_text())
data = json.loads(summary.read_text())
candidate = data["candidates"][0]["metrics"]
thresholds = prereg["public_gate"]["stage_r1_thresholds"]
counts = {
    "right_success": int(candidate["right"]["successes"]),
    "left_success": int(candidate["left"]["successes"]),
    "total_success": int(candidate["all"]["successes"]),
    "grasp_once": int(candidate["all"]["grasp_completions"]),
}
checks = {key: counts[key] >= value for key, value in thresholds.items()}
report = {
    "format": "strict-track2-v258-public-stage-r1-audit-v1",
    "counts": counts,
    "thresholds": thresholds,
    "checks": checks,
    "accepted": all(checks.values()),
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["accepted"] else 6)
