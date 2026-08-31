#!/usr/bin/env python3
"""Preregister the public-only right-arm kNN world-model feasibility audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v214_public_right_knn_action_visual_diagnostic_seed1413"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V213 = JOINT / "v213_v208_right_logit_lr5e7_long128_seed1412"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v213_report = V213 / "audit/v213_long128_logit_report.json"
    report = json.loads(v213_report.read_text())
    if report.get("passed") is not False or not (V213 / "V213_RIGHT_PARENT_REJECTED").exists():
        raise ValueError("v214 requires the completed preregistered v213 rejection")
    scripts = [
        BASE / "pipeline/scripts/build_v214_public_knn_library.py",
        BASE / "pipeline/scripts/export_v214_public_knn_cache.py",
    ]
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    prereg = {
        "format": "strict-track2-v214-public-right-knn-diagnostic-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test whether a stateless public-training-only nonparametric action-conditioned "
            "world model restores right-arm motion and separates success from failure rewards"
        ),
        "motivation": {
            "v213_report": str(v213_report),
            "v213_report_sha256": sha256(v213_report),
            "v213_step100_success_hit_rate": report["candidates"][-1]["success_reward"]["threshold_hit_rate"],
            "v213_step100_peak_max": report["candidates"][-1]["success_reward"]["peak_max"],
            "visual_finding": "parametric parent freezes/ghosts the right arm during long recursive rollout",
        },
        "fixed_model": {
            "library": "v163 mixed public training windows, right arm only",
            "query_episode_exclusion": True,
            "descriptor_visual": "16x16 RGB block mean from latest context frame",
            "descriptor_action": "normalized 4-history plus 8-future actions",
            "visual_weight": 1.0,
            "action_weight": 2.5,
            "retrieved_output": "unaltered eight target RGB frames of nearest public training window",
            "left_route": "frozen v212 baseline cache",
            "chunks": 16,
            "stateless": True,
            "outcome_labels_used_for_retrieval": False,
            "query_target_frames_read": False,
        },
        "fixed_audit": {
            "success_holdout": "same episode-disjoint public success holdout as v212/v213",
            "failure_holdout": "same episode-disjoint public on-policy failure holdout as v212/v213",
            "official_reward_threshold": 0.9,
            "feasibility_success_hit_rate_min": 0.25,
            "feasibility_failure_hit_rate_max": 0.05,
            "feasibility_hit_rate_margin_min": 0.20,
            "no_service_promotion": True,
            "no_policy_training": True,
        },
        "implementation": {str(path): sha256(path) for path in scripts},
        "guards": {
            "declared_public_training_data_only": True,
            "official_request_data": False,
            "hidden_or_final_evaluation_data": False,
            "query_outcome_labels": False,
            "policy_behavior_cloning": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "registry": str(REG), "fixed_model": prereg["fixed_model"]}, indent=2))


if __name__ == "__main__":
    main()
