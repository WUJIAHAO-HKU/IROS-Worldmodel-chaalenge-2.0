#!/usr/bin/env python3
"""Apply the preregistered v215 reward-discrimination gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def hit_summary(report: dict, mask: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(report["raw_scores"]["candidate"], dtype=np.float64)[mask]
    peaks = scores.max(1)
    return {
        "sequence_count": int(len(peaks)),
        "mean": float(scores.mean()),
        "peak_mean": float(peaks.mean()),
        "peak_max": float(peaks.max()),
        "hit_count": int((peaks >= threshold).sum()),
        "hit_rate": float((peaks >= threshold).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads((args.registry / "preregistration.json").read_text())
    with np.load(args.run / "audit/public_success_baseline.npz", allow_pickle=False) as values:
        success_right = values["arm_right"].astype(bool)
    with np.load(args.run / "audit/public_failure_baseline.npz", allow_pickle=False) as values:
        failure_right = values["arm_right"].astype(bool)
    threshold = float(prereg["fixed_audit"]["threshold"])
    candidates = []
    for fixed in prereg["fixed_candidates"]:
        tag = fixed["tag"]
        success_report = json.loads((args.run / f"audit/{tag}_public_success_reward.json").read_text())
        failure_report = json.loads((args.run / f"audit/{tag}_public_failure_reward.json").read_text())
        success = hit_summary(success_report, success_right, threshold)
        failure = hit_summary(failure_report, failure_right, threshold)
        checks = {
            "success_recall": success["hit_rate"] >= prereg["fixed_audit"]["candidate_success_hit_rate_min"],
            "failure_target_consistency": failure["hit_rate"] <= prereg["fixed_audit"]["candidate_failure_hit_rate_max"],
            "discrimination_margin": success["hit_rate"] - failure["hit_rate"] >= prereg["fixed_audit"]["candidate_discrimination_margin_min"],
        }
        candidates.append({**fixed, "success": success, "failure": failure, "checks": checks, "passed": all(checks.values())})
    passing = [candidate for candidate in candidates if candidate["passed"]]
    selected = min(passing, key=lambda item: (item["failure"]["hit_rate"], -item["success"]["hit_rate"], item["visual_weight"])) if passing else None
    report = {
        "format": "strict-track2-v215-public-right-knn-visual-sweep-v1",
        "passed": selected is not None,
        "candidates": candidates,
        "selected": selected,
        "service_promoted": False,
        "policy_training": False,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    (args.run / "audit/v215_knn_visual_sweep_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.run / ("V215_KNN_WEIGHT_FEASIBLE" if report["passed"] else "V215_KNN_WEIGHT_REJECTED")).touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
