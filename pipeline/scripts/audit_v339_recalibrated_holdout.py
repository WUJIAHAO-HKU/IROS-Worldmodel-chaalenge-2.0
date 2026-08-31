#!/usr/bin/env python3
"""Apply the train-selected v339 threshold to frozen v337 holdout probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], split: str, threshold: float) -> dict:
    selected = [row for row in rows if row["split"] == split]
    label = np.asarray([row["recursive_label"] for row in selected], dtype=bool)
    recursive = np.asarray([row["recursive_probability"] for row in selected])
    teacher = np.asarray([row["teacher_probability"] for row in selected])
    return {
        "rows": len(selected),
        "corrupt_contexts": int(label.sum()),
        "teacher_specificity": float((teacher < threshold).mean()),
        "recursive_negative_specificity": float((recursive[~label] < threshold).mean()),
        "corrupt_recall": float((recursive[label] >= threshold).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("gate", "calibration-report", "source-holdout-report", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    calibration = json.loads(args.calibration_report.read_text())
    if calibration.get("passed") is not True:
        raise RuntimeError("v339 calibration did not pass")
    source = json.loads(args.source_holdout_report.read_text())
    if source.get("format") != "strict-track2-v337-recursive-ood-holdout-gate-v1":
        raise RuntimeError("wrong frozen holdout source")
    with np.load(args.gate, allow_pickle=False) as gate:
        threshold = float(gate["threshold"].item())
    if abs(threshold - float(calibration["selected"]["threshold"])) > 1e-7:
        raise RuntimeError("v339 threshold drift")
    aggregates = {
        split: summarize(source["rows"], split, threshold)
        for split in ("validation", "local_test")
    }
    checks = {
        "source_exact_512_rows": len(source["rows"]) == 512,
        "validation_teacher_specificity_ge_0p98": aggregates["validation"]["teacher_specificity"] >= 0.98,
        "local_teacher_specificity_ge_0p98": aggregates["local_test"]["teacher_specificity"] >= 0.98,
        "validation_recursive_negative_specificity_ge_0p93": aggregates["validation"]["recursive_negative_specificity"] >= 0.93,
        "local_recursive_negative_specificity_ge_0p93": aggregates["local_test"]["recursive_negative_specificity"] >= 0.93,
        "validation_corrupt_recall_ge_0p65": aggregates["validation"]["corrupt_recall"] >= 0.65,
        "local_corrupt_recall_ge_0p65": aggregates["local_test"]["corrupt_recall"] >= 0.65,
    }
    report = {
        "format": "strict-track2-v339-recalibrated-recursive-ood-holdout-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "aggregates": aggregates,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes_reanchor_candidate_design": all(checks.values()),
        "evidence_sha256": {
            "gate": sha(args.gate),
            "calibration_report": sha(args.calibration_report),
            "source_holdout_report": sha(args.source_holdout_report),
        },
        "guards": {
            "threshold_selected_from_public_train_oof_only": True,
            "holdout_probabilities_reused_without_model_rerun": True,
            "reward_or_success_outcomes_read": False,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
