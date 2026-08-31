#!/usr/bin/env python3
"""Freeze the v364 training-mode rollout-only decision."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    prereg = json.loads(args.preregistration.read_text())
    metrics = raw["metrics"]
    trajectories = int(round(metrics["num_trajectories"]))
    rate = float(metrics["success_once"])
    successes = int(round(rate * trajectories))
    minimum = int(prereg["gate"]["minimum_successes"])
    protocol_ok = (
        raw["protocol"]["policy_updates"] == 0
        and raw["protocol"]["checkpoint_writes"] == 0
        and raw["protocol"]["rollout_epochs"] == 4
        and trajectories == 128
    )
    passed = protocol_ok and successes >= minimum
    report = {
        "format": "strict-track2-v364-trainmode-rolloutonly-go-no-go-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "successes": successes,
        "trajectories": trajectories,
        "success_fraction": f"{successes}/{trajectories}",
        "success_rate": rate,
        "minimum_successes": minimum,
        "v327_v326_baseline_successes": 14,
        "improvement_vs_v327_count": successes - 14,
        "protocol_ok": protocol_ok,
        "decision": "permit_one_conservative_update_after_disk_authorization" if passed else "reject_v355_before_rl_training",
        "metrics": metrics,
        "evidence_sha256": {"raw": sha(args.raw), "preregistration": sha(args.preregistration)},
        "guards": raw["guards"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
