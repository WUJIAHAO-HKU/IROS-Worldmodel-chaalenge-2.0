#!/usr/bin/env python3
"""Freeze the v361 rollout-only go/no-go decision from TensorBoard metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    events = sorted((args.run / "tensorboard").glob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError("no TensorBoard event files")
    values: dict[str, float] = {}
    all_tags: set[str] = set()
    for path in events:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags().get("scalars", []))
        all_tags.update(tags)
        for tag in tags:
            points = acc.Scalars(tag)
            if points:
                values[tag] = float(points[-1].value)
    success_tag = next((tag for tag in ("eval/success_once", "eval/success_rate", "eval/success") if tag in values), None)
    trajectory_tag = next((tag for tag in ("eval/num_trajectories", "eval/episodes", "eval/num_episodes") if tag in values), None)
    if success_tag is None:
        raise RuntimeError(f"no success metric; available={sorted(all_tags)}")
    trajectories = int(round(values[trajectory_tag])) if trajectory_tag else int(prereg["protocol"]["trajectories"])
    rate = values[success_tag]
    successes = int(round(rate * trajectories))
    minimum = int(prereg["gate"]["minimum_successes"])
    passed = trajectories == 32 and math.isfinite(rate) and successes >= minimum
    report = {
        "format": "strict-track2-v361-rolloutonly-go-no-go-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "success_metric": success_tag,
        "trajectory_metric": trajectory_tag,
        "successes": successes,
        "trajectories": trajectories,
        "success_fraction": f"{successes}/{trajectories}",
        "success_rate": rate,
        "minimum_successes": minimum,
        "decision": "permit_one_conservative_update_preregistration" if passed else "reject_v355_before_rl_training",
        "all_scalar_metrics": values,
        "evidence_sha256": {
            "preregistration": sha(args.preregistration),
            "event_files": {path.name: sha(path) for path in events},
        },
        "guards": {"policy_updates": 0, "checkpoint_writes": 0, "hidden_or_final_data": False, "real_submission": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
