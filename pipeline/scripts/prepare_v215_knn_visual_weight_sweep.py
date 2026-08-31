#!/usr/bin/env python3
"""Preregister a public-only kNN visual-continuity weight sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v215_public_right_knn_visual_weight_sweep_seed1414"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v214_report_path = V214 / "audit/v214_public_knn_report.json"
    v214_report = json.loads(v214_report_path.read_text())
    if v214_report.get("feasible") is not False or not (V214 / "V214_KNN_NOT_FEASIBLE").exists():
        raise ValueError("v215 requires the completed preregistered v214 rejection")
    failure_report_path = V214 / "audit/public_failure_reward.json"
    failure_report = json.loads(failure_report_path.read_text())
    with np.load(V214 / "audit/public_failure_baseline.npz", allow_pickle=False) as values:
        right = values["arm_right"].astype(bool)
    target = np.asarray(failure_report["raw_scores"]["target"], dtype=np.float64)[right]
    target_failure_hit_rate = float((target.max(1) >= 0.9).mean())
    library = V214 / "library/public_right_knn.npz"
    exporter = BASE / "pipeline/scripts/export_v214_public_knn_cache.py"
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    prereg = {
        "format": "strict-track2-v215-public-right-knn-visual-sweep-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "reduce v214 public failure false positives through stronger stateless visual-continuity matching",
        "inputs": {
            "v214_report": str(v214_report_path),
            "v214_report_sha256": sha256(v214_report_path),
            "library": str(library),
            "library_sha256": sha256(library),
            "exporter": str(exporter),
            "exporter_sha256": sha256(exporter),
        },
        "fixed_candidates": [
            {"tag": "visual05", "visual_weight": 5.0, "action_weight": 2.5},
            {"tag": "visual10", "visual_weight": 10.0, "action_weight": 2.5},
            {"tag": "visual25", "visual_weight": 25.0, "action_weight": 2.5},
        ],
        "fixed_audit": {
            "threshold": 0.9,
            "target_failure_hit_rate": target_failure_hit_rate,
            "failure_hit_rate_tolerance": 0.05,
            "candidate_failure_hit_rate_max": target_failure_hit_rate + 0.05,
            "candidate_success_hit_rate_min": 0.75,
            "candidate_discrimination_margin_min": 0.45,
            "selection_order": "lowest failure hit rate, highest success hit rate, lowest visual weight",
            "no_service_promotion": True,
            "no_policy_training": True,
        },
        "guards": {
            "same_public_holdouts_as_v214": True,
            "retrieval_uses_outcome_labels": False,
            "query_target_frames_read": False,
            "hidden_or_final_evaluation_data": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "fixed_candidates": prereg["fixed_candidates"], "fixed_audit": prereg["fixed_audit"]}, indent=2))


if __name__ == "__main__":
    main()
