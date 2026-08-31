#!/usr/bin/env python3
"""Preregister a non-disruptive training-only continuation gate at step 6."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
    off = root / "artifacts/strict_track2_official_20260810"
    name = "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
    reg = off / "run_registry" / name
    output = reg / "global_step6_continuation_gate_preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    auditor = root / "pipeline/scripts/audit_v318_step6_stop_gate.py"
    watcher = root / "pipeline/scripts/watch_v318_step6_stop_gate.sh"
    for path in (auditor, watcher, reg / "preregistration.json"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v318-global-step6-continuation-gate-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "decision_point": {"checkpoint": "global_step_6", "metric_steps": [3, 4, 5]},
        "thresholds": {
            "steps3_5_success_once_sum_min": 0.15625,
            "steps3_5_return_mean_min": 0.09,
            "step5_success_once_min": 0.03125,
        },
        "actions": {
            "pass": "continue uninterrupted to the fixed global_step_10 budget",
            "fail": "reject v318 and stop before spending steps 6 through 9",
        },
        "tools": {
            "auditor": {"path": str(auditor), "sha256": sha(auditor)},
            "watcher": {"path": str(watcher), "sha256": sha(watcher)},
        },
        "guards": {
            "registered_before_step4_metrics": True,
            "training_metrics_only": True,
            "public112_outcomes_read": False,
            "final128_outcomes_read": False,
            "hidden_outcomes_read": False,
            "real_submission": False,
            "algorithm_or_budget_changed": False,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
