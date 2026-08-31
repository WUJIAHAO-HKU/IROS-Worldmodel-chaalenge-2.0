#!/usr/bin/env python3
"""Freeze gate for the V15.8 hybrid-source, arm-routed Track 2 parent."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onpolicy-visual", required=True, type=Path)
    parser.add_argument("--official-demo", required=True, type=Path)
    parser.add_argument("--reward-alignment", required=True, type=Path)
    parser.add_argument("--prior-arm-screen", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--candidate-release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    visual = json.loads(args.onpolicy_visual.read_text())
    demo = json.loads(args.official_demo.read_text())
    reward = json.loads(args.reward_alignment.read_text())
    prior = json.loads(args.prior_arm_screen.read_text())
    prereg = json.loads(args.preregistration.read_text())
    raw = {name: np.asarray(value) for name, value in reward["raw_scores"].items()}
    baseline_cache = Path(reward["reuse_baseline_cache"] or reward["cache"])
    with np.load(baseline_cache, allow_pickle=False) as values:
        right = values["arm_right"].astype(bool)
    terminal_corr = {
        "overall_baseline": corr(raw["baseline"][:, -1], raw["target"][:, -1]),
        "overall_candidate": corr(raw["candidate"][:, -1], raw["target"][:, -1]),
        "right_baseline": corr(raw["baseline"][right, -1], raw["target"][right, -1]),
        "right_candidate": corr(raw["candidate"][right, -1], raw["target"][right, -1]),
    }
    routing = visual["routing"]
    improvement = visual["improvement"]
    reward_groups = reward["groups"]
    checks = {
        "preregistered_format": prereg.get("format")
        == "strict-track2-v158-hybrid-arm-routed-parent-preregistration-v1",
        "prior_arm_experts_screen_passed": prior.get("passed") is True,
        "onpolicy_candidate_routes_64": int(routing["candidate"]) == 64,
        "onpolicy_left_and_right_experts_used": int(routing["candidate_left"]) > 0
        and int(routing["candidate_right"]) > 0,
        "official_demo_baseline_routes_16": int(demo["routing"]["baseline"]) == 16,
        "official_demo_bit_exact_16": int(demo["routing"]["bit_exact_to_baseline"]) == 16,
        "visual_overall_rgb_ge_20": improvement["overall"]["rgb_mae_improvement_percent"] >= 20.0,
        "visual_right_contact_ge_25": improvement["right"]["contact_rgb_mae_improvement_percent"] >= 25.0,
        "visual_success_rgb_ge_15": improvement["capture_success"]["rgb_mae_improvement_percent"] >= 15.0,
        "visual_left_rgb_ge_15": improvement["left"]["rgb_mae_improvement_percent"] >= 15.0,
        "visual_texture_nonregression": improvement["overall"]["texture_mae_improvement_percent"] >= 0.0,
        "visual_temporal_nonregression": improvement["overall"]["temporal_delta_mae_improvement_percent"] >= 0.0,
        "reward_overall_mae_nonregression": reward_groups["overall"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "reward_right_mae_nonregression": reward_groups["right"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "reward_overall_terminal_nonregression": reward_groups["overall"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_right_terminal_nonregression": reward_groups["right"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_left_terminal_nonregression": reward_groups["left"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_success_terminal_nonregression": reward_groups["capture_success"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_overall_terminal_pearson_nonregression": terminal_corr["overall_candidate"] >= terminal_corr["overall_baseline"],
        "reward_right_terminal_pearson_nonregression": terminal_corr["right_candidate"] >= terminal_corr["right_baseline"],
    }
    evidence_paths = {
        "onpolicy_visual": args.onpolicy_visual,
        "official_demo": args.official_demo,
        "reward_alignment": args.reward_alignment,
        "prior_arm_screen": args.prior_arm_screen,
        "preregistration": args.preregistration,
    }
    report = {
        "format": "strict-track2-v158-hybrid-arm-routed-parent-screen-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "candidate_release": str(args.candidate_release.resolve()),
        "terminal_pearson": terminal_corr,
        "evidence": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in evidence_paths.items()
        },
        "track2_guards": {
            "policy_actions_modified": False,
            "participant_action_selection": False,
            "mpc": False,
            "official_pi05_reward_rl_modified": False,
            "real_policy_development_or_acceptance_used_for_parent_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
