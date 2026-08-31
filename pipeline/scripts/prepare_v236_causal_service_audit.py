#!/usr/bin/env python3
"""Preregister the v236 causal right-close retrieval service audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v236_v235c_causal_right_close_service_seed1435_20260818"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V235 = OFF / "run_registry/v235c_public_causal_retrieval_ranking_seed1434_20260818"
MODEL_VERSION = "track2-v236-public-knn-alpha070-causal-right-close"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    ranking_path = V235 / "sweep_report.json"
    ranking = json.loads(ranking_path.read_text())
    selected = next(
        row for row in ranking["candidates"] if row["name"] == "causal070_raw"
    )
    if selected["right_expert_similarity_terminal_correlation"] < 0.10:
        raise ValueError("v235c causal candidate lacks expert-action ranking")
    if selected["right_motion_terminal_correlation"] < 0.08:
        raise ValueError("v235c causal candidate lacks positive right-motion ranking")
    library = V214 / "library/public_right_knn.npz"
    source_paths = [
        BASE / "pipeline/wam_pipeline/v236_causal_public_knn_runtime.py",
        BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py",
        BASE / "pipeline/wam_pipeline/backends.py",
        BASE / "pipeline/wam_pipeline/service.py",
        BASE / "pipeline/scripts/export_v217_service_recursive_cache.py",
        BASE / "pipeline/scripts/summarize_v236_causal_service.py",
        BASE / "pipeline/scripts/restart_v236_services.sh",
        BASE / "pipeline/scripts/launch_v236_causal_service_audit.sh",
    ]
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    (RUN / "local_dev_token.txt").write_text("local-dev-token\n")
    prereg = {
        "format": "strict-track2-v236-causal-service-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "retain public right-arm retrieval only on frames whose requested "
            "right gripper action is physically closed"
        ),
        "inputs": {
            "parent_release": str(V209),
            "parent_manifest_sha256": sha256(
                V209 / "arm_routed_autoregressive_manifest.json"
            ),
            "public_library": str(library),
            "public_library_sha256": sha256(library),
            "v235c_ranking": str(ranking_path),
            "v235c_ranking_sha256": sha256(ranking_path),
        },
        "fixed_model": {
            "model_version": MODEL_VERSION,
            "left_route": "bit-identical frozen v209 left parent",
            "right_parent": "frozen v209 right parent",
            "right_retrieval": "v218 nearest public window, visual=1.0/action=2.5",
            "right_alpha_per_frame": "0.70 iff requested action[13] < 0.5 else 0",
            "route_inputs": ["instruction", "history_actions", "future_actions"],
            "service_output": "eight future RGB frames only",
        },
        "fixed_audit": {
            "chunks": 16,
            "threshold": 0.9,
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.20,
            "hit_rate_margin_min": 0.55,
            "expert_similarity_terminal_correlation_min": 0.10,
            "right_motion_terminal_correlation_min": 0.08,
            "right_group_terminal_std_min": 1e-7,
            "official_http_acceptance_required": True,
            "no_policy_training": True,
        },
        "implementation": {str(path): sha256(path) for path in source_paths},
        "guards": {
            "public_data_only": True,
            "outcomes_or_rewards_used_at_runtime": False,
            "request_id_or_seed_used_for_retrieval": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    release = {
        "format": "strict-track2-v236-causal-service-release-registration-v1",
        "model_version": MODEL_VERSION,
        "parent_release": str(V209),
        "library_index": str(library),
        "alpha": 0.70,
        "causal_gate": "future_actions[:,13] < 0.5",
        "source_sha256": prereg["implementation"],
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    (RUN / "release_registration.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
