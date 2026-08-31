#!/usr/bin/env python3
"""Final screen for an action-gated parent under visual and official-reward gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onpolicy-visual", required=True)
    parser.add_argument("--official-demo", required=True)
    parser.add_argument("--reward-alignment", required=True)
    parser.add_argument("--blend-selection", required=True)
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "onpolicy_visual": args.onpolicy_visual,
            "official_demo": args.official_demo,
            "reward_alignment": args.reward_alignment,
            "blend_selection": args.blend_selection,
        }.items()
    }
    visual = json.loads(paths["onpolicy_visual"].read_text())
    demo = json.loads(paths["official_demo"].read_text())
    reward = json.loads(paths["reward_alignment"].read_text())
    selection = json.loads(paths["blend_selection"].read_text())
    raw = {name: np.asarray(value) for name, value in reward["raw_scores"].items()}
    with np.load(reward["reuse_baseline_cache"] or reward["cache"], allow_pickle=False) as values:
        right = values["arm_right"].astype(bool)
    terminal_corr = {
        "overall_baseline": corr(raw["baseline"][:, -1], raw["target"][:, -1]),
        "overall_candidate": corr(raw["candidate"][:, -1], raw["target"][:, -1]),
        "right_baseline": corr(raw["baseline"][right, -1], raw["target"][right, -1]),
        "right_candidate": corr(raw["candidate"][right, -1], raw["target"][right, -1]),
    }
    vi = visual["improvement"]
    ri = reward["groups"]
    checks = {
        "blend_selection_passed": selection.get("passed") is True,
        "selected_blend_is_0p12": abs(float(selection["selected"]["strength"]) - 0.12) < 1e-12,
        "onpolicy_routes_candidate_64": int(visual["routing"]["candidate"]) == 64,
        "demo_routes_baseline_16": int(demo["routing"]["baseline"]) == 16,
        "demo_bit_exact_16": int(demo["routing"]["bit_exact_to_baseline"]) == 16,
        "visual_overall_rgb_ge_3": vi["overall"]["rgb_mae_improvement_percent"] >= 3.0,
        "visual_right_contact_ge_3": vi["right"]["contact_rgb_mae_improvement_percent"] >= 3.0,
        "visual_left_rgb_nonregression": vi["left"]["rgb_mae_improvement_percent"] >= 0.0,
        "visual_texture_nonregression": vi["overall"]["texture_mae_improvement_percent"] >= 0.0,
        "visual_temporal_nonregression": vi["overall"]["temporal_delta_mae_improvement_percent"] >= 0.0,
        "reward_overall_mae_nonregression": ri["overall"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "reward_right_mae_nonregression": ri["right"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "reward_overall_terminal_mae_nonregression": ri["overall"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_right_terminal_mae_nonregression": ri["right"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_left_terminal_mae_nonregression": ri["left"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_success_terminal_mae_nonregression": ri["capture_success"]["improvement"]["terminal_gain_mae_improvement_percent"] >= 0.0,
        "reward_overall_terminal_pearson_nonregression": terminal_corr["overall_candidate"] >= terminal_corr["overall_baseline"],
        "reward_right_terminal_pearson_nonregression": terminal_corr["right_candidate"] >= terminal_corr["right_baseline"],
    }
    report = {
        "format": "strict-track2-reward-safe-parent-screen-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "candidate_release": str(Path(args.candidate_release).resolve()),
        "selected_blend": float(selection["selected"]["strength"]),
        "terminal_pearson": terminal_corr,
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "track2_guards": {
            "policy_actions_modified": False,
            "participant_action_selection": False,
            "mpc": False,
            "real_policy_development_or_acceptance_used": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
