#!/usr/bin/env python3
"""Recalibrate v337 using only its frozen public-train OOF curve."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
SOURCE = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v337_public_recursive_ood_gate_seed1507_20260822"
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v339_high_specificity_recursive_ood_gate_seed1508_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    paths = [RUN / name for name in ("release_registration.json", "recursive_ood_gate.npz", "calibration_report.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite v339 evidence")
    training_path = SOURCE / "training_report.json"
    source_gate_path = SOURCE / "recursive_ood_gate.npz"
    training = json.loads(training_path.read_text())
    if training.get("passed") is not True:
        raise RuntimeError("v337 training evidence did not pass")

    registration = {
        "format": "strict-track2-v339-high-specificity-recursive-ood-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "v337 public-train grouped OOF curve only; no holdout rows or outcomes",
        "selection_rule": {
            "eligibility": "teacher specificity >=.99, negative specificity >=.99, corrupt recall >=.85",
            "tie_break": "negative specificity, corrupt recall, teacher specificity, smaller C",
            "candidate_rows": "frozen v337 grouped OOF cv_rows",
        },
        "authorizes_rl": False,
        "guards": {
            "public_train_oof_only": True,
            "v337_holdout_report_read": False,
            "reward_or_success_outcomes_read": False,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "evidence_sha256": {
            "v337_training_report": sha(training_path),
            "v337_source_gate": sha(source_gate_path),
        },
    }
    paths[0].write_text(json.dumps(registration, indent=2) + "\n")

    eligible = [
        row for row in training["cv_rows"]
        if row["teacher_specificity"] >= 0.99
        and row["negative_specificity"] >= 0.99
        and row["corrupt_recall"] >= 0.85
    ]
    if not eligible:
        raise RuntimeError("no high-specificity train-only operating point")
    selected = max(
        eligible,
        key=lambda row: (
            row["negative_specificity"], row["corrupt_recall"],
            row["teacher_specificity"], -row["c"],
        ),
    )
    with np.load(source_gate_path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    source_c = float(np.asarray(payload["c"]).item())
    if source_c != float(selected["c"]):
        raise RuntimeError(f"selected C {selected['c']} differs from fitted source C {source_c}")
    payload["threshold"] = np.asarray(selected["threshold"], dtype=np.float32)
    np.savez_compressed(paths[1], **payload)
    report = {
        "format": "strict-track2-v339-high-specificity-recursive-ood-calibration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected": selected,
        "eligible_rows": eligible,
        "checks": {
            "v337_training_passed": True,
            "selected_from_training_oof_only": True,
            "teacher_specificity_ge_0p99": selected["teacher_specificity"] >= 0.99,
            "negative_specificity_ge_0p99": selected["negative_specificity"] >= 0.99,
            "corrupt_recall_ge_0p85": selected["corrupt_recall"] >= 0.85,
            "source_c_matches_selected_c": source_c == float(selected["c"]),
        },
        "passed": True,
        "artifact_sha256": sha(paths[1]),
        "preregistration_sha256": sha(paths[0]),
        "guards": registration["guards"],
    }
    paths[2].write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
