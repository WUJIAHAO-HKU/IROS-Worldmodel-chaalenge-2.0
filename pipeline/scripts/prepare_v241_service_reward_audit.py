#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v241b_v240_alignment_gated_transport_service_seed1441_20260819"
RUN, REG = JOINT / NAME, OFF / "run_registry" / NAME
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V240 = OFF / "run_registry/v240_public_alignment_gated_transport_seed1439_20260819"
MODEL_VERSION = "track2-v241-alignment-gated-right-transport"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    analysis_path = V240 / "analysis_report.json"
    analysis = json.loads(analysis_path.read_text())
    selected = analysis.get("selected")
    expected = {
        "library_path_quantile": 0.9, "temperature": 10.0,
        "alignment_floor": 0.5, "motion_quantile": 0.75,
    }
    if selected is None or any(selected.get(k) != v for k, v in expected.items()):
        raise ValueError("v240 did not select the frozen v241 alignment gate")
    if min(
        selected["alpha_expert_similarity_correlation"],
        selected["alpha_motion_correlation"],
        selected["alpha_directional_alignment_correlation"],
    ) <= 0:
        raise ValueError("v240 selected gate does not have all-positive correlations")
    library = V214 / "library/public_right_knn.npz"
    sources = [
        BASE / "pipeline/wam_pipeline/v241_alignment_gated_transport_runtime.py",
        BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py",
        BASE / "pipeline/wam_pipeline/backends.py",
        BASE / "pipeline/wam_pipeline/service.py",
        BASE / "pipeline/scripts/replay_v241_post_grasp_reward.py",
        BASE / "pipeline/scripts/restart_v241_services.sh",
        BASE / "pipeline/scripts/launch_v241_service_reward_audit.sh",
    ]
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    (RUN / "local_dev_token.txt").write_text("local-dev-token\n")
    prereg = {
        "format": "strict-track2-v241-service-reward-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "reward only expert-aligned, nontrivial post-grasp right-arm transport",
        "inputs": {"parent_release": str(V209), "public_library": str(library),
                   "public_library_sha256": sha256(library),
                   "v240_analysis": str(analysis_path), "v240_analysis_sha256": sha256(analysis_path)},
        "fixed_model": {"model_version": MODEL_VERSION, **expected,
                        "distance_scale": 26.628942489624023,
                        "motion_scale": selected["motion_scale"],
                        "service_output": "eight future RGB frames only"},
        "fixed_capture_gate": {"post_grasp_queries_min": 64,
                               "post_grasp_group_std_mean_min": 1e-5,
                               "post_grasp_nonzero_group_fraction_min": 0.25,
                               "post_grasp_motion_reward_correlation_min": 0.05,
                               "official_http_acceptance_required": True},
        "implementation": {str(path): sha256(path) for path in sources},
        "guards": {"public_data_only": True, "outcomes_or_rewards_used_at_runtime": False,
                   "request_id_or_seed_used_for_retrieval": False,
                   "policy_modified": False, "hidden_or_final_data": False,
                   "real_submission": False},
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    release = {
        "format": "strict-track2-v241-alignment-gated-transport-release-v1",
        "model_version": MODEL_VERSION, "parent_release": str(V209),
        "library_index": str(library), "parameters": prereg["fixed_model"],
        "source_sha256": prereg["implementation"],
        "hidden_or_final_data": False, "real_submission": False,
    }
    (RUN / "release_registration.json").write_text(json.dumps(release, indent=2) + "\n")


if __name__ == "__main__":
    main()
