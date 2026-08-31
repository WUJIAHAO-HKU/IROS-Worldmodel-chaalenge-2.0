#!/usr/bin/env python3
"""Preregister public-only parametric/retrieval blend coefficients."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v216_public_right_parametric_knn_blend_sweep_seed1415"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V215 = JOINT / "v215_public_right_knn_visual_weight_sweep_seed1414"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v215_report_path = V215 / "audit/v215_knn_visual_sweep_report.json"
    v215 = json.loads(v215_report_path.read_text())
    if v215.get("passed") is not False or not (V215 / "V215_KNN_WEIGHT_REJECTED").exists():
        raise ValueError("v216 requires completed preregistered v215 rejection")
    success_retrieval = V214 / "audit/public_success_candidate.npz"
    failure_retrieval = V214 / "audit/public_failure_candidate.npz"
    blender = BASE / "pipeline/scripts/blend_v216_retrieval_cache.py"
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    candidates = [
        {"tag": "alpha050", "alpha": 0.50},
        {"tag": "alpha070", "alpha": 0.70},
        {"tag": "alpha085", "alpha": 0.85},
        {"tag": "alpha095", "alpha": 0.95},
    ]
    prereg = {
        "format": "strict-track2-v216-public-parametric-knn-blend-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "retain public retrieval success dynamics while suppressing transient failure reward excursions",
        "inputs": {
            "v215_report": str(v215_report_path),
            "v215_report_sha256": sha256(v215_report_path),
            "success_retrieval_cache": str(success_retrieval),
            "success_retrieval_sha256": sha256(success_retrieval),
            "failure_retrieval_cache": str(failure_retrieval),
            "failure_retrieval_sha256": sha256(failure_retrieval),
            "blender": str(blender),
            "blender_sha256": sha256(blender),
        },
        "fixed_candidates": candidates,
        "fixed_audit": {
            "threshold": 0.9,
            "candidate_success_hit_rate_min": 0.75,
            "candidate_failure_hit_rate_max": 0.32272727272727275,
            "candidate_discrimination_margin_min": 0.45,
            "selection_order": "lowest failure hit rate, highest success hit rate, lowest alpha",
            "no_service_promotion": True,
            "no_policy_training": True,
        },
        "guards": {
            "blend_uses_labels_or_reward": False,
            "same_public_holdouts_as_v214_v215": True,
            "hidden_or_final_evaluation_data": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "fixed_candidates": candidates, "fixed_audit": prereg["fixed_audit"]}, indent=2))


if __name__ == "__main__":
    main()
