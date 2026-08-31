#!/usr/bin/env python3
"""Apply preregistered same-domain start-zero parent promotion gates."""

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


def visual(report: dict, group: str, metric: str) -> float:
    return float(report["improvement"][group][f"{metric}_improvement_percent"])


def reward(report: dict, group: str, metric: str) -> float:
    return float(report["groups"][group]["improvement"][f"{metric}_improvement_percent"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--official-visual", required=True)
    parser.add_argument("--mixed-visual", required=True)
    parser.add_argument("--official-reward", required=True)
    parser.add_argument("--mixed-reward", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value).resolve() for name, value in {
            "candidate": args.candidate,
            "preregistration": args.preregistration,
            "official_visual": args.official_visual,
            "mixed_visual": args.mixed_visual,
            "official_reward": args.official_reward,
            "mixed_reward": args.mixed_reward,
        }.items()
    }
    prereg = json.loads(paths["preregistration"].read_text())
    official_visual = json.loads(paths["official_visual"].read_text())
    mixed_visual = json.loads(paths["mixed_visual"].read_text())
    official_reward = json.loads(paths["official_reward"].read_text())
    mixed_reward = json.loads(paths["mixed_reward"].read_text())
    minimum_reward = float(
        prereg["parent_promotion_gates"]["official_validation_reward_mae_improvement_percent_min"]
    )
    checks = [
        ("official.overall.rgb_mae", visual(official_visual, "overall", "rgb_mae"), 0.0),
        ("official.left.rgb_mae", visual(official_visual, "left", "rgb_mae"), 0.0),
        ("official.right.rgb_mae", visual(official_visual, "right", "rgb_mae"), 0.0),
        ("official.overall.motion_rgb_mae", visual(official_visual, "overall", "motion_rgb_mae"), 0.0),
        ("official.overall.contact_rgb_mae", visual(official_visual, "overall", "contact_rgb_mae"), 0.0),
        ("official.overall.reward_mae", reward(official_reward, "overall", "reward_mae"), minimum_reward),
        ("official.left.reward_mae", reward(official_reward, "left", "reward_mae"), 0.0),
        ("official.right.reward_mae", reward(official_reward, "right", "reward_mae"), 0.0),
        ("mixed.synthetic.rgb_mae", visual(mixed_visual, "synthetic", "rgb_mae"), 0.0),
        ("mixed.synthetic.reward_mae", reward(mixed_reward, "synthetic", "reward_mae"), 0.0),
        ("mixed.left.rgb_mae", visual(mixed_visual, "left", "rgb_mae"), 0.0),
        ("mixed.right.rgb_mae", visual(mixed_visual, "right", "rgb_mae"), 0.0),
        ("mixed.left.reward_mae", reward(mixed_reward, "left", "reward_mae"), 0.0),
        ("mixed.right.reward_mae", reward(mixed_reward, "right", "reward_mae"), 0.0),
    ]
    records = [
        {
            "name": name,
            "value_percent": measured,
            "minimum_percent": minimum,
            "passed": measured >= minimum,
        }
        for name, measured, minimum in checks
    ]
    diagnostics = {
        "official.overall.temporal_delta_mae_improvement_percent": visual(
            official_visual, "overall", "temporal_delta_mae"
        ),
        "official.right.terminal_gain_mae_improvement_percent": reward(
            official_reward, "right", "terminal_gain_mae"
        ),
        "mixed.synthetic.temporal_delta_mae_improvement_percent": visual(
            mixed_visual, "synthetic", "temporal_delta_mae"
        ),
        "mixed.right.terminal_gain_mae_improvement_percent": reward(
            mixed_reward, "right", "terminal_gain_mae"
        ),
    }
    report = {
        "format": "strict-track2-initial-window-parent-screen-v1",
        "passed": all(record["passed"] for record in records),
        "candidate_checkpoint": str(paths["candidate"]),
        "checks": records,
        "failed_checks": [record["name"] for record in records if not record["passed"]],
        "non_gate_diagnostics": diagnostics,
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
            if name != "candidate"
        },
        "interpretation_guard": "Passing promotes a world-model parent to official fixed-budget RL; it is not real-RoboTwin policy success.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
