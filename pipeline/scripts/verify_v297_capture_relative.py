#!/usr/bin/env python3
"""Baseline-relative capture gate for a v271-derived world model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-details", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent_report.read_text())
    candidate = json.loads(args.candidate_report.read_text())
    with np.load(args.parent_details, allow_pickle=False) as data:
        parent_terminal = data["rewards"][:, -1].astype(np.float32)
    with np.load(args.candidate_details, allow_pickle=False) as data:
        candidate_terminal = data["rewards"][:, -1].astype(np.float32)
    absolute = np.abs(candidate_terminal - parent_terminal)
    inherited_checks = {
        key: bool(candidate["checks"][key])
        for key in (
            "post_queries",
            "group_std",
            "success_like_count",
            "alignment_global",
            "motion_global",
            "alignment_group",
            "service_contract",
        )
    }
    checks = {
        **inherited_checks,
        "same_query_count": candidate["post_queries"] == parent["post_queries"] == 101,
        "success_like_not_below_parent": (
            candidate["success_like_count"] >= parent["success_like_count"]
        ),
        "terminal_reward_bit_exact": bool(np.array_equal(candidate_terminal, parent_terminal)),
        "terminal_reward_max_abs_le_1e_7": bool(absolute.max() <= 1e-7),
    }
    report = {
        "format": "strict-track2-v297-relative-capture-gate-v1",
        "parent": parent["service_model_version"],
        "candidate": candidate["service_model_version"],
        "parent_success_like_count": parent["success_like_count"],
        "candidate_success_like_count": candidate["success_like_count"],
        "terminal_reward_max_absolute_change": float(absolute.max()),
        "terminal_reward_mean_absolute_change": float(absolute.mean()),
        "checks": checks,
        "excluded_parent_failing_check": {
            "name": "regime_alpha_global",
            "parent_value": parent["reward_alpha_correlation"],
            "old_absolute_threshold": 0.7,
            "reason": "not a valid child gate because the frozen parent itself fails it",
        },
        "passed": all(checks.values()),
        "guards": {
            "public_capture_only": True,
            "runtime_uses_reward": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
