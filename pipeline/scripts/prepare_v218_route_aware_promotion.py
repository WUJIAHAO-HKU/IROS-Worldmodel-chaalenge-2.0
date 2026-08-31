#!/usr/bin/env python3
"""Preregister the unchanged-pixel v218 route-aware service promotion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v218_v217_route_aware_service_promotion_seed1417"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V217 = JOINT / "v217_v216_online_recursive_service_gate_seed1416"
MODEL_VERSION = "track2-v218-public-knn-blend-alpha070-route-aware"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v217_report_path = V217 / "audit/v217_online_recursive_report.json"
    route_path = V217 / "audit/deployable_route_mismatch_diagnostic.json"
    v217 = json.loads(v217_report_path.read_text())
    route = json.loads(route_path.read_text())
    substantive = route["v217_substantive_checks"]
    if v217.get("passed") is not False or not all(substantive.values()):
        raise ValueError("v218 requires v217 rejection solely after substantive gates passed")
    if route["groups"]["public_failure_mismatched"]["count"] <= 0:
        raise ValueError("v218 requires the deployable-route mismatch diagnosis")
    source_paths = [
        BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py",
        BASE / "pipeline/wam_pipeline/backends.py",
        BASE / "pipeline/wam_pipeline/service.py",
        BASE / "pipeline/scripts/audit_v218_real_context_determinism.py",
        BASE / "pipeline/scripts/summarize_v218_route_aware_promotion.py",
        BASE / "pipeline/scripts/restart_v218_services.sh",
        BASE / "pipeline/scripts/launch_v218_route_aware_promotion.sh",
    ]
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    prereg = {
        "format": "strict-track2-v218-route-aware-service-promotion-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "promote unchanged v217 pixels using only deployable instruction/action route semantics",
        "inputs": {
            "v217_report": str(v217_report_path),
            "v217_report_sha256": sha256(v217_report_path),
            "route_diagnostic": str(route_path),
            "route_diagnostic_sha256": sha256(route_path),
            "parent_release": str(V209),
            "public_library": str(V214 / "library/public_right_knn.npz"),
        },
        "fixed_model": {
            "model_version": MODEL_VERSION,
            "pixels_and_runtime_identical_to_v217": True,
            "route_inputs": ["instruction", "history_actions", "future_actions"],
            "route_prohibited_inputs": ["arm_right audit label", "reward", "success", "request_id", "evaluation result"],
            "right_blend_alpha": 0.70,
            "service_output": "eight future RGB frames only",
        },
        "fixed_audit": {
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.32272727272727275,
            "hit_rate_margin_min": 0.45,
            "route_matched_first_chunk_max_abs": 4,
            "real_context_repeat_pixel_exact": True,
            "action_permutation_changed_count_min": 1,
            "no_policy_training": True,
        },
        "implementation": {str(path): sha256(path) for path in source_paths},
        "guards": {
            "model_selected_from_hidden_or_final_data": False,
            "audit_labels_available_to_service": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    release = {
        "format": "strict-track2-v218-route-aware-service-release-registration-v1",
        "model_version": MODEL_VERSION,
        "parent_release": str(V209),
        "library_index": str(V214 / "library/public_right_knn.npz"),
        "alpha": 0.70,
        "visual_weight": 1.0,
        "action_weight": 2.5,
        "source_sha256": prereg["implementation"],
        "pixel_runtime_sha256": sha256(BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py"),
        "hidden_or_final_evaluation_data": False,
        "real_submission": False,
    }
    (RUN / "release_registration.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "fixed_model": prereg["fixed_model"], "fixed_audit": prereg["fixed_audit"]}, indent=2))


if __name__ == "__main__":
    main()
