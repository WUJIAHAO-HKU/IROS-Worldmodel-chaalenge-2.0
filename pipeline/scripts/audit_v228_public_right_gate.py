#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys
from datetime import datetime, timezone


METRICS = ("success_once", "left_success", "right_success", "grasp_once", "right_grasp", "arm_left", "arm_right", "num_trajectories")


def extract(text: str, name: str) -> float:
    for pattern in (rf"'eval/{re.escape(name)}': array\(([-+0-9.eE]+)", rf"'eval/{re.escape(name)}': ([-+0-9.eE]+)"):
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    raise ValueError(f"missing eval/{name}")


prereg = json.loads(pathlib.Path(sys.argv[1]).read_text())
log_path, output_path = map(pathlib.Path, sys.argv[2:4])
values = {name: extract(log_path.read_text(errors="replace"), name) for name in METRICS}
total = int(round(values["num_trajectories"]))
counts = {
    "total_success": int(round(values["success_once"] * total)),
    "left_success": int(round(values["left_success"] * total)),
    "right_success": int(round(values["right_success"] * total)),
    "grasp_once": int(round(values["grasp_once"] * total)),
    "right_grasp": int(round(values["right_grasp"] * total)),
    "left_episodes": int(round(values["arm_left"] * total)),
    "right_episodes": int(round(values["arm_right"] * total)),
    "num_trajectories": total,
}
assert counts["left_episodes"] == 4 and counts["right_episodes"] == 12
thresholds = (
    prereg["thresholds"]
    if "thresholds" in prereg
    else prereg["public_gate"]["thresholds"]
)
checks = {key: counts[key] >= minimum for key, minimum in thresholds.items()}
report = {
    "audited_at": datetime.now(timezone.utc).isoformat(),
    "counts": counts,
    "thresholds": thresholds,
    "checks": checks,
    "accepted": all(checks.values()),
    "real_submission": False,
    "reserved_final128_used": False,
}
output_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["accepted"] else 4)
