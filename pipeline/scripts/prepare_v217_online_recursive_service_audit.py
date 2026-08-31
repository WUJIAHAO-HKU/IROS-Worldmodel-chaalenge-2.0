#!/usr/bin/env python3
"""Preregister the v217 real-service recursive public gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v217_v216_online_recursive_service_gate_seed1416"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V216 = JOINT / "v216_public_right_parametric_knn_blend_sweep_seed1415"
MODEL_VERSION = "track2-v217-public-knn-blend-alpha070-online-gated"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v216_report_path = V216 / "audit/v216_blend_sweep_report.json"
    v216 = json.loads(v216_report_path.read_text())
    selected = v216.get("selected") or {}
    if not v216.get("passed") or selected.get("alpha") != 0.70:
        raise ValueError("v217 requires the completed v216 alpha=0.70 selection")
    library = V214 / "library/public_right_knn.npz"
    source_paths = [
        BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py",
        BASE / "pipeline/wam_pipeline/backends.py",
        BASE / "pipeline/wam_pipeline/service.py",
        BASE / "pipeline/scripts/export_v217_service_recursive_cache.py",
        BASE / "pipeline/scripts/summarize_v217_online_recursive.py",
        BASE / "pipeline/scripts/restart_v217_services.sh",
        BASE / "pipeline/scripts/launch_v217_online_recursive_audit.sh",
    ]
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    (RUN / "local_dev_token.txt").write_text("local-dev-token\n")
    prereg = {
        "format": "strict-track2-v217-online-recursive-service-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "verify v216 alpha=0.70 under the actual stateless HTTP service and recursively blended contexts",
        "inputs": {
            "v216_run": str(V216),
            "v216_report": str(v216_report_path),
            "v216_report_sha256": sha256(v216_report_path),
            "parent_release": str(V209),
            "parent_manifest_sha256": sha256(V209 / "arm_routed_autoregressive_manifest.json"),
            "public_library": str(library),
            "public_library_sha256": sha256(library),
            "public_library_manifest_sha256": sha256(library.with_suffix(".manifest.json")),
        },
        "fixed_model": {
            "model_version": MODEL_VERSION,
            "left_route": "frozen v209 left parent",
            "right_parent": "frozen v209 right parent",
            "right_retrieval": "nearest public-training window using visual=1.0/action=2.5",
            "right_blend_alpha": 0.70,
            "recursive_context": "previous blended service RGB",
            "retrieval_uses_outcomes_or_rewards": False,
            "service_output": "eight future RGB frames only",
        },
        "fixed_audit": {
            "success_holdout": "same episode-disjoint public success holdout as v216",
            "failure_holdout": "same episode-disjoint public on-policy failure holdout as v216",
            "chunks": 16,
            "threshold": 0.9,
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.32272727272727275,
            "hit_rate_margin_min": 0.45,
            "first_chunk_must_match_v216_bit_exact": True,
            "official_http_acceptance_required": True,
            "no_policy_training": True,
        },
        "implementation": {str(path): sha256(path) for path in source_paths},
        "guards": {
            "declared_public_training_data_only": True,
            "request_id_or_seed_used_for_retrieval": False,
            "hidden_or_final_evaluation_data": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    release = {
        "format": "strict-track2-v217-public-knn-blend-release-registration-v1",
        "model_version": MODEL_VERSION,
        "parent_release": str(V209),
        "library_index": str(library),
        "alpha": 0.70,
        "visual_weight": 1.0,
        "action_weight": 2.5,
        "source_sha256": prereg["implementation"],
        "hidden_or_final_evaluation_data": False,
        "real_submission": False,
    }
    (RUN / "release_registration.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "registry": str(REG), "fixed_model": prereg["fixed_model"]}, indent=2))


if __name__ == "__main__":
    main()
