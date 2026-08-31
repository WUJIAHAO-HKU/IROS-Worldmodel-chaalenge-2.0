#!/usr/bin/env python3
"""Preregister a public-data-only Cartesian/image-pose reconstruction pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v373_cartesian_pose_progress_seed1536_20260823"
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
    trainer = ROOT / "pipeline/scripts/train_action_pose_v170.py"
    model = ROOT / "pipeline/wam_pipeline/object_geometry_v170.py"
    split = ROOT / "artifacts/splits/adjust_bottle_50episodes_full.json"
    dataset = ROOT / "artifacts/datasets/aloha-agilex_clean_50/data"
    for path in (trainer, model, split, dataset):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v373-cartesian-pose-progress-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "reconstruct action-conditioned end-effector image geometry from supplied public "
            "trajectories, then test a continuous right-arm physical progress feature before "
            "authorizing any new world-model training"
        ),
        "fixed_reconstruction": {
            "source": "supplied 50 public RoboTwin trajectories only",
            "train_split": "official train40 minus frozen dev episodes 36 and 47",
            "dev_episodes": [36, 47],
            "target": "next-frame projected end-effector origin/axes/depth for both arms",
            "steps": 2200,
            "batch_size": 512,
            "learning_rate": 0.002,
            "hidden": 256,
            "seed": 1536,
        },
        "fixed_acceptance": {
            "worst_arm_landmark_mae_px_max": 1.5,
            "right_origin_mae_px_max": 1.5,
            "right_depth_mae_m_max": 0.003,
            "all_checks_required": True,
        },
        "next_stage_if_passed": (
            "read-only physical-progress audit on frozen policy rollout action captures; "
            "no reward/outcome labels used to fit or tune the pose model"
        ),
        "evidence_sha256": {
            str(path): sha256(path) for path in (trainer, model, split)
        },
        "guards": {
            "participant_component": "world-model RGB dynamics only",
            "public_world_model_supervision_only": True,
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REGISTRY.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REGISTRY / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REGISTRY)}, indent=2))


if __name__ == "__main__":
    main()
