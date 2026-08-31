#!/usr/bin/env python3
"""Select a preregistered reward-safe world-model blend strength."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def tag(strength: float) -> str:
    return f"alpha_{strength:.4f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--reports-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preregistration_path = Path(args.preregistration).resolve()
    preregistration = json.loads(preregistration_path.read_text())
    root = Path(args.reports_directory).resolve()
    gates = preregistration["required_gates"]
    rows = []
    for strength_value in preregistration["screen_strengths"]:
        strength = float(strength_value)
        stem = tag(strength)
        visual_path = root / f"{stem}_visual.json"
        reward_path = root / f"{stem}_reward.json"
        visual = json.loads(visual_path.read_text())
        reward = json.loads(reward_path.read_text())
        raw = {name: np.asarray(value) for name, value in reward["raw_scores"].items()}
        with np.load(reward["reuse_baseline_cache"] or reward["cache"], allow_pickle=False) as values:
            right = values["arm_right"].astype(bool)
        terminal = {
            "overall_baseline": corr(raw["baseline"][:, -1], raw["target"][:, -1]),
            "overall_candidate": corr(raw["candidate"][:, -1], raw["target"][:, -1]),
            "right_baseline": corr(raw["baseline"][right, -1], raw["target"][right, -1]),
            "right_candidate": corr(raw["candidate"][right, -1], raw["target"][right, -1]),
        }
        visual_improvement = visual["improvement"]
        reward_improvement = reward["groups"]
        checks = {
            "overall_rgb": visual_improvement["overall"]["rgb_mae_improvement_percent"]
            >= float(gates["overall_rgb_improvement_percent_min"]),
            "right_contact_rgb": visual_improvement["right"]["contact_rgb_mae_improvement_percent"]
            >= float(gates["right_contact_rgb_improvement_percent_min"]),
            "left_rgb": visual_improvement["left"]["rgb_mae_improvement_percent"]
            >= float(gates["left_rgb_improvement_percent_min"]),
            "overall_terminal_reward_mae": reward_improvement["overall"]["improvement"]["terminal_gain_mae_improvement_percent"]
            >= float(gates["overall_terminal_reward_mae_improvement_percent_min"]),
            "right_terminal_reward_mae": reward_improvement["right"]["improvement"]["terminal_gain_mae_improvement_percent"]
            >= float(gates["right_terminal_reward_mae_improvement_percent_min"]),
            "overall_terminal_reward_pearson": terminal["overall_candidate"] >= terminal["overall_baseline"],
            "right_terminal_reward_pearson": terminal["right_candidate"] >= terminal["right_baseline"],
        }
        rows.append(
            {
                "strength": strength,
                "visual_report": str(visual_path),
                "visual_report_sha256": sha256(visual_path),
                "reward_report": str(reward_path),
                "reward_report_sha256": sha256(reward_path),
                "visual_improvement": {
                    "overall_rgb_percent": visual_improvement["overall"]["rgb_mae_improvement_percent"],
                    "right_contact_percent": visual_improvement["right"]["contact_rgb_mae_improvement_percent"],
                    "left_rgb_percent": visual_improvement["left"]["rgb_mae_improvement_percent"],
                },
                "reward_improvement": {
                    "overall_mae_percent": reward_improvement["overall"]["improvement"]["reward_mae_improvement_percent"],
                    "right_mae_percent": reward_improvement["right"]["improvement"]["reward_mae_improvement_percent"],
                    "overall_terminal_mae_percent": reward_improvement["overall"]["improvement"]["terminal_gain_mae_improvement_percent"],
                    "right_terminal_mae_percent": reward_improvement["right"]["improvement"]["terminal_gain_mae_improvement_percent"],
                },
                "terminal_pearson": terminal,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    eligible = [row for row in rows if row["passed"]]
    selected = max(eligible, key=lambda row: row["strength"]) if eligible else None
    report = {
        "format": "strict-track2-terminal-reward-blend-selection-v1",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": sha256(preregistration_path),
        "passed": selected is not None,
        "selected": selected,
        "candidates": rows,
        "requires_exact_runtime_confirmation": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
