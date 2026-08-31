#!/usr/bin/env python3
"""Persist a same-metric, training-only v318/v308 step-0 comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    "env/return",
    "env/reward",
    "env/success_once",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/grad_norm",
    "train/actor/kl_loss",
)


def value(report: dict, metric: str) -> float:
    return float(report["metrics"][metric]["0"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v318", required=True, type=Path)
    parser.add_argument("--v308", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    current = json.loads(args.v318.read_text())
    previous = json.loads(args.v308.read_text())
    if current.get("passed") is not True or previous.get("passed") is not True:
        raise RuntimeError("both source prefix audits must pass")
    rows = {}
    for metric in METRICS:
        new = value(current, metric)
        old = value(previous, metric)
        rows[metric] = {
            "v318": new,
            "v308": old,
            "difference": new - old,
            "ratio": new / old if old != 0 else None,
        }
    report = {
        "format": "strict-track2-v318-v308-training-step0-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "same_metric_same_step": True,
        "comparison": rows,
        "observations": {
            "reward_signal_strength_improved": rows["env/return"]["ratio"] > 1.0 and rows["env/reward"]["ratio"] > 1.0,
            "reward_coverage_improved": rows["env/success_once"]["difference"] > 0.0,
            "v318_safety_prefix_passed": current["passed"] is True,
            "single_step_proves_policy_improvement": False,
        },
        "selection_scope": "training-only diagnostic; no public, reserved-final, hidden, or submission result",
        "guards": {
            "policy_or_optimizer_modified": False,
            "world_model_or_reward_modified": False,
            "training_interrupted": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
