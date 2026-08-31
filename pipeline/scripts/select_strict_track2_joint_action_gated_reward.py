#!/usr/bin/env python3
"""Apply the preregistered terminal-reward gates to joint parent checkpoints."""

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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--demo-report", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("STEP", "CHECKPOINT", "VISUAL_REPORT", "REWARD_REPORT"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preregistration_path = Path(args.preregistration).resolve()
    preregistration = json.loads(preregistration_path.read_text())
    gates = preregistration["required_gates"]
    demo_path = Path(args.demo_report).resolve()
    demo = json.loads(demo_path.read_text())
    expected_demo = int(gates["official_demo_bit_exact_count"])
    demo_bit_exact = int(demo["routing"]["bit_exact_to_baseline"])

    rows = []
    for raw_step, raw_checkpoint, raw_visual, raw_reward in args.candidate:
        step = int(raw_step)
        checkpoint = Path(raw_checkpoint).resolve()
        visual_path = Path(raw_visual).resolve()
        reward_path = Path(raw_reward).resolve()
        visual = json.loads(visual_path.read_text())
        reward = json.loads(reward_path.read_text())
        improvement = reward["groups"]
        raw = {name: np.asarray(value) for name, value in reward["raw_scores"].items()}
        with np.load(reward["reuse_baseline_cache"] or reward["cache"], allow_pickle=False) as values:
            right = values["arm_right"].astype(bool)
        target_terminal = raw["target"][:, -1]
        baseline_terminal = raw["baseline"][:, -1]
        candidate_terminal = raw["candidate"][:, -1]
        terminal_corr = {
            "overall_baseline": correlation(baseline_terminal, target_terminal),
            "overall_candidate": correlation(candidate_terminal, target_terminal),
            "right_baseline": correlation(baseline_terminal[right], target_terminal[right]),
            "right_candidate": correlation(candidate_terminal[right], target_terminal[right]),
        }
        checks = {
            "official_demo_bit_exact": demo_bit_exact == expected_demo,
            "onpolicy_overall_rgb": visual["improvement"]["overall"]["rgb_mae_improvement_percent"]
            >= float(gates["onpolicy_overall_rgb_improvement_percent_min"]),
            "onpolicy_right_contact": visual["improvement"]["right"]["contact_rgb_mae_improvement_percent"]
            >= float(gates["onpolicy_right_contact_rgb_improvement_percent_min"]),
            "onpolicy_left_rgb": visual["improvement"]["left"]["rgb_mae_improvement_percent"]
            >= float(gates["onpolicy_left_rgb_improvement_percent_min"]),
            "reward_overall_mae": improvement["overall"]["improvement"]["reward_mae_improvement_percent"]
            >= float(gates["official_reward_overall_mae_improvement_percent_min"]),
            "reward_right_mae": improvement["right"]["improvement"]["reward_mae_improvement_percent"]
            >= float(gates["official_reward_right_mae_improvement_percent_min"]),
            "reward_overall_terminal_mae": improvement["overall"]["improvement"]["terminal_gain_mae_improvement_percent"]
            >= float(gates["official_reward_terminal_mae_improvement_percent_min"]),
            "reward_right_terminal_mae": improvement["right"]["improvement"]["terminal_gain_mae_improvement_percent"]
            >= float(gates["official_reward_right_terminal_mae_improvement_percent_min"]),
            "reward_terminal_pearson": terminal_corr["overall_candidate"]
            >= terminal_corr["overall_baseline"],
        }
        row = {
            "step": step,
            "checkpoint": str(checkpoint),
            "visual_report": str(visual_path),
            "visual_report_sha256": sha256(visual_path),
            "reward_report": str(reward_path),
            "reward_report_sha256": sha256(reward_path),
            "terminal_pearson": terminal_corr,
            "checks": checks,
            "passed": all(checks.values()),
            "ranking_tuple": [
                improvement["right"]["improvement"]["terminal_gain_mae_improvement_percent"],
                improvement["overall"]["improvement"]["terminal_gain_mae_improvement_percent"],
                visual["improvement"]["right"]["contact_rgb_mae_improvement_percent"],
            ],
        }
        rows.append(row)

    eligible = [row for row in rows if row["passed"]]
    selected = max(eligible, key=lambda row: tuple(row["ranking_tuple"])) if eligible else None
    output = {
        "format": "strict-track2-joint-action-gated-reward-selection-v1",
        "preregistration": str(preregistration_path),
        "preregistration_sha256": sha256(preregistration_path),
        "demo_report": str(demo_path),
        "demo_report_sha256": sha256(demo_path),
        "passed": selected is not None,
        "selected": selected,
        "candidates": rows,
        "interpretation_guard": "No real RoboTwin development or acceptance result is used here.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
