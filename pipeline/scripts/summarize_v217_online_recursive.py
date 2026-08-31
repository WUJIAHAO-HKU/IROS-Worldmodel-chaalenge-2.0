#!/usr/bin/env python3
"""Apply preregistered gates to the v217 online recursive service audit."""

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
    with np.load(args.run / "audit/public_success_baseline.npz", allow_pickle=False) as values:
        success_right = values["arm_right"].astype(bool)
    with np.load(args.run / "audit/public_failure_baseline.npz", allow_pickle=False) as values:
        failure_right = values["arm_right"].astype(bool)
    fixed = prereg["fixed_audit"]
    threshold = float(fixed["threshold"])
    success = summarize(args.run / "audit/public_success_reward.json", success_right, threshold)
    failure = summarize(args.run / "audit/public_failure_reward.json", failure_right, threshold)
    v216 = Path(prereg["inputs"]["v216_run"])
    first_chunk = {}
    first_chunk_exact = True
    for tag in ("public_success", "public_failure"):
        with np.load(args.run / f"audit/{tag}_candidate.npz", allow_pickle=False) as values:
            actual = values["candidate"][:, :8].copy()
        with np.load(v216 / f"audit/alpha070_{tag}_candidate.npz", allow_pickle=False) as values:
            expected = values["candidate"][:, :8].copy()
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        exact = bool(np.array_equal(actual, expected))
        first_chunk[tag] = {
            "pixel_exact": exact,
            "pixel_mae": float(difference.mean()),
            "pixel_max_abs": int(difference.max()),
        }
        first_chunk_exact = first_chunk_exact and exact
    checks = {
        "success_recall": success["hit_rate"] >= fixed["success_hit_rate_min"],
        "failure_target_consistency": failure["hit_rate"] <= fixed["failure_hit_rate_max"],
        "discrimination_margin": success["hit_rate"] - failure["hit_rate"] >= fixed["hit_rate_margin_min"],
        "service_contract": (args.run / "audit/service_acceptance.json").is_file(),
        "first_chunk_api_equivalence": first_chunk_exact,
    }
    report = {
        "format": "strict-track2-v217-online-recursive-service-gate-v1",
        "passed": all(checks.values()),
        "success": success,
        "failure": failure,
        "checks": checks,
        "first_chunk_api_equivalence": first_chunk,
        "model_version": prereg["fixed_model"]["model_version"],
        "policy_training": False,
        "hidden_or_final_data": False,
        "real_submission": False,
        "final_128_used": False,
    }
    output = args.run / "audit/v217_online_recursive_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    (args.run / ("V217_ONLINE_GATE_PASSED" if report["passed"] else "V217_ONLINE_GATE_REJECTED")).touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
