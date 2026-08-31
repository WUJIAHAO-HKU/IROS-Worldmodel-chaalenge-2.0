#!/usr/bin/env python3
"""Apply the preregistered public long-horizon gates to v236."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path, mask: np.ndarray, threshold: float) -> dict:
    report = json.loads(path.read_text())
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
    with np.load(args.run / "audit/public_success_baseline.npz", allow_pickle=False) as item:
        success_right = item["arm_right"].astype(bool)
    with np.load(args.run / "audit/public_failure_baseline.npz", allow_pickle=False) as item:
        failure_right = item["arm_right"].astype(bool)
    fixed = prereg["fixed_audit"]
    threshold = float(fixed["threshold"])
    success = summarize(args.run / "audit/public_success_reward.json", success_right, threshold)
    failure = summarize(args.run / "audit/public_failure_reward.json", failure_right, threshold)
    ranking = json.loads(Path(prereg["inputs"]["v235c_ranking"]).read_text())
    selected = next(row for row in ranking["candidates"] if row["name"] == "causal070_raw")
    checks = {
        "success_recall": success["hit_rate"] >= fixed["success_hit_rate_min"],
        "failure_consistency": failure["hit_rate"] <= fixed["failure_hit_rate_max"],
        "discrimination_margin": success["hit_rate"] - failure["hit_rate"] >= fixed["hit_rate_margin_min"],
        "expert_action_ranking": selected["right_expert_similarity_terminal_correlation"] >= fixed["expert_similarity_terminal_correlation_min"],
        "positive_right_motion_ranking": selected["right_motion_terminal_correlation"] >= fixed["right_motion_terminal_correlation_min"],
        "within_group_variation": selected["right_group_terminal_std"]["mean"] >= fixed["right_group_terminal_std_min"],
        "service_contract": (args.run / "audit/service_acceptance.json").is_file(),
    }
    report = {
        "format": "strict-track2-v236-causal-service-gate-v1",
        "passed": all(checks.values()),
        "success": success,
        "failure": failure,
        "ranking": selected,
        "checks": checks,
        "model_version": prereg["fixed_model"]["model_version"],
        "policy_training": False,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    output = args.run / "audit/v236_causal_service_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (args.run / ("V236_PUBLIC_GATE_PASSED" if report["passed"] else "V236_PUBLIC_GATE_REJECTED")).touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
