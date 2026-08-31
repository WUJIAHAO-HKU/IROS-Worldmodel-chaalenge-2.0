#!/usr/bin/env python3
"""Preregister a higher-step-size public-only right-world-model pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v213_v208_right_logit_lr5e7_long128_seed1412"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V208 = JOINT / "v208_v205_mixed_right_gripper_contrast_long32_seed1407"
V212 = JOINT / "v212_v208_right_logit_long128_seed1411"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {NAME}")
    v208_report_path = V208 / "audit/v208_right_contrast_report.json"
    v208_report = json.loads(v208_report_path.read_text())
    right_parent = V208 / "selected_right_expert"
    if v208_report.get("passed") is not True or v208_report["selected"]["step"] != 1200:
        raise ValueError("frozen v208 right parent is not accepted at step 1200")
    if sha256(right_parent / "model.pt") != v208_report["selected_model_sha256"]:
        raise ValueError("v208 selected right model hash mismatch")

    v212_report_path = V212 / "audit/v212_long128_logit_report.json"
    v212_report = json.loads(v212_report_path.read_text())
    if v212_report.get("passed") is not False:
        raise ValueError("v213 is only justified after preregistered v212 rejection")
    if not (V212 / "V212_RIGHT_PARENT_REJECTED").exists():
        raise ValueError("v212 rejection marker is missing")
    v212_prereg_path = OFF / "run_registry/v212_v208_right_logit_long128_seed1411/preregistration.json"
    v212_prereg = json.loads(v212_prereg_path.read_text())

    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    trainer = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
    prereg = {
        "format": "strict-track2-v213-right-logit-lr5e7-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test whether a tenfold effective optimizer step and success oversampling "
            "can recover absolute right-arm reward over 128 recursive public frames"
        ),
        "diagnosis": {
            "v212_report": str(v212_report_path),
            "v212_report_sha256": sha256(v212_report_path),
            "v212_result": "rejected by preregistered long128 gates",
            "v212_learning_rate": v212_prereg["fixed_training"]["learning_rate"],
            "finding": (
                "v212 remained effectively unchanged because gradients were clipped while "
                "the optimizer learning rate was only 5e-8"
            ),
        },
        "parent": str(right_parent),
        "parent_model_sha256": sha256(right_parent / "model.pt"),
        "trainer": str(trainer),
        "trainer_sha256": sha256(trainer),
        "data": v212_prereg["data"],
        "fixed_training": {
            "steps": 100,
            "checkpoint_steps": [25, 50, 75, 100],
            "batch_size": 1,
            "chunks": 16,
            "frames_per_sequence": 128,
            "arm_filter": "right",
            "learning_rate": 5e-7,
            "reward_objective": "logit",
            "reward_loss_weight": 0.02,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 4.0,
            "success_weight": 10.0,
            "late_start": 64,
            "late_weight": 4.0,
            "terminal_visual_weight": 2.0,
            "max_grad_norm": 1.0,
            "seed": 1412,
        },
        "fixed_audit": {
            "reuse_v212_frozen_baseline": True,
            "success_holdout": "same episode-disjoint public 10-episode split as v212",
            "failure_holdout": "same episode-disjoint public on-policy failure split as v212",
            "recursive_chunks": 16,
            "candidate_order": [25, 50, 75, 100],
            "release_requires_absolute_reward_gate": True,
            "absolute_success_threshold": 0.9,
            "right_success_visual_mae_ratio_max": 1.03,
            "right_failure_visual_mae_ratio_max": 1.03,
            "right_failure_false_positive_rate_max": 0.0,
        },
        "guards": {
            "world_model_training_only": True,
            "public_training_and_holdout_data_only": True,
            "frozen_official_reward_model": True,
            "fixed_official_policy_unchanged": True,
            "policy_behavior_cloning": False,
            "policy_action_injection": False,
            "real_submission": False,
            "final_128_used": False,
            "hidden_or_official_request_data": False,
        },
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "registry": str(REG), "training": prereg["fixed_training"]}, indent=2))


if __name__ == "__main__":
    main()
