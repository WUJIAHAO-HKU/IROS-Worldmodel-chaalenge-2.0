#!/usr/bin/env python3
"""Apply preregistered long128 visual, reward and absolute-threshold gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_abs(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(a[mask].astype(np.float32) - b[mask].astype(np.float32)).mean())


def score_summary(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float | int]:
    selected = np.asarray(scores, dtype=np.float64)[mask]
    peaks = selected.max(axis=1)
    terminal = selected[:, -1]
    return {
        "sequences": int(selected.shape[0]),
        "mean": float(selected.mean()),
        "peak_mean": float(peaks.mean()),
        "peak_max": float(peaks.max()),
        "terminal_mean": float(terminal.mean()),
        "threshold_hit_rate": float((peaks >= threshold).mean()),
        "threshold_hit_count": int((peaks >= threshold).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prereg = json.loads((args.registry / "preregistration.json").read_text())
    threshold = float(prereg["fixed_audit"]["absolute_success_threshold"])
    success_base_path = args.run / "audit/baseline/public_success_long128.npz"
    failure_base_path = args.run / "audit/baseline/public_failure_long128.npz"
    success_base_report = json.loads(
        (args.run / "audit/baseline/public_success_reward.json").read_text()
    )
    failure_base_report = json.loads(
        (args.run / "audit/baseline/public_failure_reward.json").read_text()
    )
    with np.load(success_base_path, allow_pickle=False) as values:
        success_target = values["target"].copy()
        success_baseline = values["baseline"].copy()
        success_right = values["arm_right"].astype(bool)
    with np.load(failure_base_path, allow_pickle=False) as values:
        failure_target = values["target"].copy()
        failure_baseline = values["baseline"].copy()
        failure_right = values["arm_right"].astype(bool)
    if not success_right.any() or not failure_right.any():
        raise ValueError("long128 holdouts must contain right-arm sequences")

    baseline_success_scores = np.asarray(
        success_base_report["raw_scores"]["baseline"], dtype=np.float64
    )
    target_success_scores = np.asarray(
        success_base_report["raw_scores"]["target"], dtype=np.float64
    )
    baseline_failure_scores = np.asarray(
        failure_base_report["raw_scores"]["baseline"], dtype=np.float64
    )
    baseline_success_visual = mean_abs(success_baseline, success_target, success_right)
    baseline_failure_visual = mean_abs(failure_baseline, failure_target, failure_right)
    baseline_success_summary = score_summary(
        baseline_success_scores, success_right, threshold
    )
    target_success_summary = score_summary(target_success_scores, success_right, threshold)
    baseline_failure_summary = score_summary(
        baseline_failure_scores, failure_right, threshold
    )

    candidates = []
    for step in prereg["fixed_audit"]["candidate_order"]:
        audit = args.run / "audit" / f"step{int(step):04d}"
        success_report = json.loads((audit / "public_success_reward.json").read_text())
        failure_report = json.loads((audit / "public_failure_reward.json").read_text())
        with np.load(audit / "public_success_candidate.npz", allow_pickle=False) as values:
            success_candidate = values["candidate"].copy()
        with np.load(audit / "public_failure_candidate.npz", allow_pickle=False) as values:
            failure_candidate = values["candidate"].copy()
        candidate_success_visual = mean_abs(
            success_candidate, success_target, success_right
        )
        candidate_failure_visual = mean_abs(
            failure_candidate, failure_target, failure_right
        )
        success_scores = np.asarray(
            success_report["raw_scores"]["candidate"], dtype=np.float64
        )
        failure_scores = np.asarray(
            failure_report["raw_scores"]["candidate"], dtype=np.float64
        )
        success_summary = score_summary(success_scores, success_right, threshold)
        failure_summary = score_summary(failure_scores, failure_right, threshold)
        success_visual_ratio = candidate_success_visual / baseline_success_visual
        failure_visual_ratio = candidate_failure_visual / baseline_failure_visual
        success_reward_ratio = (
            success_report["groups"]["right"]["candidate"]["reward_mae"]
            / success_report["groups"]["right"]["baseline"]["reward_mae"]
        )
        required_success_hit_rate = min(
            0.25, float(target_success_summary["threshold_hit_rate"])
        )
        checks = {
            "success_visual_nonregression": success_visual_ratio
            <= prereg["fixed_audit"]["right_success_visual_mae_ratio_max"],
            "failure_visual_nonregression": failure_visual_ratio
            <= prereg["fixed_audit"]["right_failure_visual_mae_ratio_max"],
            "success_reward_mae_improves": success_reward_ratio <= 1.0,
            "success_absolute_threshold_recall": success_summary["threshold_hit_rate"]
            >= required_success_hit_rate
            and success_summary["threshold_hit_rate"]
            > baseline_success_summary["threshold_hit_rate"],
            "failure_false_positive_nonregression": failure_summary["threshold_hit_rate"]
            <= baseline_failure_summary["threshold_hit_rate"]
            and failure_summary["threshold_hit_rate"]
            <= prereg["fixed_audit"]["right_failure_false_positive_rate_max"],
        }
        candidates.append(
            {
                "step": int(step),
                "success_visual_mae": candidate_success_visual,
                "success_visual_ratio": success_visual_ratio,
                "failure_visual_mae": candidate_failure_visual,
                "failure_visual_ratio": failure_visual_ratio,
                "success_reward_mae_ratio": success_reward_ratio,
                "success_reward": success_summary,
                "failure_reward": failure_summary,
                "required_success_hit_rate": required_success_hit_rate,
                "checks": checks,
                "failed_checks": [key for key, value in checks.items() if not value],
                "passed": all(checks.values()),
            }
        )

    passing = [item for item in candidates if item["passed"]]
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda item: (
                -item["success_reward"]["threshold_hit_rate"],
                item["success_reward_mae_ratio"],
                item["success_visual_ratio"] + item["failure_visual_ratio"],
                item["step"],
            ),
        )
        source = args.run / "checkpoints" / f"checkpoint_step_{selected['step']:06d}"
        destination = args.run / "selected_right_expert"
        if destination.exists():
            raise ValueError("refusing to overwrite selected_right_expert")
        shutil.copytree(source, destination)
        selected["model_sha256"] = sha256(destination / "model.pt")

    report = {
        "format": "strict-track2-v212-long128-logit-audit-v1",
        "passed": selected is not None,
        "threshold": threshold,
        "baseline": {
            "success_visual_mae": baseline_success_visual,
            "failure_visual_mae": baseline_failure_visual,
            "success_reward": baseline_success_summary,
            "failure_reward": baseline_failure_summary,
        },
        "target_success_reward": target_success_summary,
        "candidates": candidates,
        "selected": selected,
        "selection_inputs": "episode-disjoint public success and public on-policy failure world-model holdouts only",
        "hidden_or_final_evaluation_data": False,
        "policy_evaluation": False,
        "real_submission": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    marker = args.run / (
        "V212_RIGHT_PARENT_ACCEPTED" if selected is not None else "V212_RIGHT_PARENT_REJECTED"
    )
    marker.touch()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
