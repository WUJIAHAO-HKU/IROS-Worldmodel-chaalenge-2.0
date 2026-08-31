#!/usr/bin/env python3
"""Summarize preregistered v214 success/failure reward feasibility gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(report_path: Path, mask: np.ndarray, threshold: float) -> dict:
    report = json.loads(report_path.read_text())
    baseline = np.asarray(report["raw_scores"]["baseline"], dtype=np.float64)[mask]
    candidate = np.asarray(report["raw_scores"]["candidate"], dtype=np.float64)[mask]
    result = {}
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        peaks = values.max(1)
        result[name] = {
            "sequence_count": int(values.shape[0]),
            "mean": float(values.mean()),
            "peak_mean": float(peaks.mean()),
            "peak_max": float(peaks.max()),
            "threshold_hit_count": int((peaks >= threshold).sum()),
            "threshold_hit_rate": float((peaks >= threshold).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads((args.registry / "preregistration.json").read_text())
    threshold = float(prereg["fixed_audit"]["official_reward_threshold"])
    masks = {}
    for split in ("success", "failure"):
        cache = args.run / "audit" / f"public_{split}_candidate.npz"
        baseline_cache = args.run / "audit" / f"public_{split}_baseline.npz"
        with np.load(baseline_cache, allow_pickle=False) as values:
            masks[split] = values["arm_right"].astype(bool)
    success = summary(args.run / "audit/public_success_reward.json", masks["success"], threshold)
    failure = summary(args.run / "audit/public_failure_reward.json", masks["failure"], threshold)
    success_rate = success["candidate"]["threshold_hit_rate"]
    failure_rate = failure["candidate"]["threshold_hit_rate"]
    checks = {
        "success_recall": success_rate >= prereg["fixed_audit"]["feasibility_success_hit_rate_min"],
        "failure_false_positive": failure_rate <= prereg["fixed_audit"]["feasibility_failure_hit_rate_max"],
        "discrimination_margin": success_rate - failure_rate >= prereg["fixed_audit"]["feasibility_hit_rate_margin_min"],
    }
    report = {
        "format": "strict-track2-v214-public-right-knn-diagnostic-v1",
        "feasible": all(checks.values()),
        "threshold": threshold,
        "success": success,
        "failure": failure,
        "checks": checks,
        "next_step_only_if_feasible": "develop continuity-preserving service candidate and full visual gates",
        "service_promoted": False,
        "policy_training": False,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    output = args.run / "audit/v214_public_knn_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (args.run / ("V214_KNN_FEASIBLE" if report["feasible"] else "V214_KNN_NOT_FEASIBLE")).touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
