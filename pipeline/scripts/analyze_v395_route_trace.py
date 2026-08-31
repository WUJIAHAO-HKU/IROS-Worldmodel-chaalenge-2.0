#!/usr/bin/env python3
"""Analyze the one-shot v394 policy-distribution preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v395-v169-v394-route-trace32-preregistration-v1":
        raise RuntimeError("wrong v395 preregistration")
    lines = args.trace.read_text().splitlines()
    rows = []
    for line in lines:
        rows.extend(json.loads(line)["batch"])
    raw = json.loads(args.raw.read_text())
    metrics = raw["metrics"]
    trajectories = int(round(metrics["num_trajectories"]))
    successes = int(round(float(metrics["success_once"]) * trajectories))

    def count(key: str) -> int:
        return sum(bool(row[key]) for row in rows)

    sequential = {}
    alive = list(rows)
    for key in ("right", "source_ready", "post_grasp", "probability_ready"):
        alive = [row for row in alive if row[key]]
        sequential[key] = len(alive)
    alive = [row for row in alive if row["failure_signature"] is None]
    sequential["no_failure_signature"] = len(alive)
    sequential["hard_phase_ready"] = sum(row["hard_phase_ready"] for row in alive)
    phase_values = np.asarray(
        [
            row["continuous_phase_probability"]
            for row in alive
            if row["continuous_phase_probability"] is not None
        ],
        dtype=np.float64,
    )
    sequential["continuous_phase_ready"] = sum(
        row["continuous_phase_ready"] for row in alive
    )
    sequential["route"] = sum(row["route"] for row in alive)
    protocol_ok = bool(
        raw["protocol"]["policy_updates"] == 0
        and raw["protocol"]["checkpoint_writes"] == 0
        and raw["protocol"]["rollout_epochs"] == 1
        and trajectories == 32
        and len(rows) == 800
    )
    minimum_successes = int(prereg["gate"]["minimum_successes"])
    minimum_routes = int(prereg["gate"]["minimum_continuous_routes"])
    passed = bool(
        protocol_ok
        and successes >= minimum_successes
        and sequential["route"] >= minimum_routes
        and sequential["continuous_phase_ready"] > sequential["hard_phase_ready"]
    )
    report = {
        "format": "strict-track2-v395-v394-route-trace-analysis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "successes": successes,
        "trajectories": trajectories,
        "success_fraction": f"{successes}/{trajectories}",
        "success_rate": float(metrics["success_once"]),
        "minimum_successes": minimum_successes,
        "requests": len(rows),
        "batches": len(lines),
        "minimum_continuous_routes": minimum_routes,
        "unconditional_counts": {
            key: count(key)
            for key in (
                "right",
                "source_ready",
                "post_grasp",
                "probability_ready",
                "hard_phase_ready",
                "continuous_phase_ready",
                "route",
            )
        },
        "sequential_survivors": sequential,
        "continuous_phase_probability_quantiles": {
            str(q): float(np.quantile(phase_values, q))
            for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)
        },
        "protocol_ok": protocol_ok,
        "comparisons": {
            "delta_successes_vs_v387_v385_same32": successes - 7,
            "extra_continuous_vs_hard_phase_routes": sequential["continuous_phase_ready"]
            - sequential["hard_phase_ready"],
        },
        "decision": "permit_one_conservative_rl_update" if passed else "reject_v394_before_rl",
        "rollout_metrics": metrics,
        "evidence_sha256": {
            "trace": sha256(args.trace),
            "raw": sha256(args.raw),
            "preregistration": sha256(args.preregistration),
        },
        "guards": raw["guards"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
