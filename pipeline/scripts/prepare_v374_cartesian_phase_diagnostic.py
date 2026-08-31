#!/usr/bin/env python3
"""Preregister the v374 Cartesian phase-aligned successor diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v374_cartesian_phase_successor_diagnostic_seed1537_20260823"
RUN = JOINT / NAME
REGISTRY = OFFICIAL / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REGISTRY.exists():
        raise FileExistsError("refusing overwrite")
    audit = ROOT / "pipeline/scripts/audit_v374_cartesian_phase_diagnostic.py"
    pose = JOINT / "v373_cartesian_pose_progress_seed1536_20260823/model/best.pt"
    pose_gate = JOINT / "v373_cartesian_pose_progress_seed1536_20260823/audit/pose_reconstruction_acceptance.json"
    split = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
    library = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for path in (audit, pose, pose_gate, split, library):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not json.loads(pose_gate.read_text())["passed"]:
        raise RuntimeError("v373 pose gate did not pass")
    payload = {
        "format": "strict-track2-v374-cartesian-phase-diagnostic-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test whether frozen action-to-pose geometry can align each requested future frame "
            "to a monotonic phase in a visually matched public expert trajectory"
        ),
        "fixed_protocol": {
            "fit_data": "public train40 only through frozen v373 and frozen v214 library",
            "evaluation": "all four untouched public right-arm holdout episodes",
            "episode_route": "nearest train40 visual descriptor from request context",
            "anchor_search_radius_rows": 32,
            "future_search_radius_rows": 48,
            "mapping": "monotonic nearest projected end-effector pose",
            "baselines": ["visual-library temporal row", "copy-last RGB"],
        },
        "fixed_acceptance": {
            "exact_holdout_episodes": [6, 7, 18, 22],
            "cartesian_phase_mae_ratio_le": 0.90,
            "cartesian_terminal_phase_mae_ratio_le": 0.90,
            "cartesian_rgb_mae_ratio_le": 0.98,
            "cartesian_delta_rgb_mae_ratio_le": 0.98,
            "all_checks_required": True,
        },
        "evidence_sha256": {str(path): sha256(path) for path in (audit, pose, pose_gate, split, library)},
        "guards": {
            "public_world_model_data_only": True,
            "reward_or_success_outcomes_used": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REGISTRY / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REGISTRY)}, indent=2))


if __name__ == "__main__":
    main()
