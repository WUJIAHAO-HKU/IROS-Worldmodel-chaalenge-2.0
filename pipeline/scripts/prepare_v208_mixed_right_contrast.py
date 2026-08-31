#!/usr/bin/env python3
"""Preregister long-horizon mixed success/failure right-arm parent training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFF = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v208_v205_mixed_right_gripper_contrast_long32_seed1407"
RUN = JOINT / TAG
REG = OFF / "run_registry" / TAG
V205 = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405"
PARENT = V205 / "selected_right_expert"
MIXED = JOINT / "v163_mixed_reward_windows"
MIXED_SPLIT = MIXED / "split_manifest.json"
PUBLIC_SPLIT = V205 / "public_demo_split.json"
V207_RUN = OFF / "runs/v207_v206_conservativekl_h200_r2_step8_lr5e6_beta005_seed1406_20260818"
DIAGNOSIS = V207_RUN / "audit/v207_right_gripper_diagnosis.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise SystemExit(f"refusing to overwrite existing {TAG}")
    v207_acceptance = json.loads((V207_RUN / "audit/public_policy_stage_a_acceptance.json").read_text())
    if v207_acceptance.get("passed") is not False:
        raise ValueError("v207 public Stage A has not been authoritatively rejected")
    diagnosis = json.loads(DIAGNOSIS.read_text())
    if diagnosis.get("passed") is not True:
        raise ValueError("v207 public action diagnosis did not pass")
    v205_audit = json.loads((V205 / "audit/right_terminal_parent_report.json").read_text())
    if v205_audit.get("passed") is not True or sha256(PARENT / "model.pt") != v205_audit["selected_model_sha256"]:
        raise ValueError("v205 selected right expert is not intact")
    mixed = json.loads(MIXED_SPLIT.read_text())
    if mixed.get("real_development_or_acceptance_data") is not False:
        raise ValueError("mixed training manifest does not exclude evaluation data")
    if mixed.get("linked_official_success_windows", 0) < 1900 or mixed.get("linked_onpolicy_train_windows", 0) < 4600:
        raise ValueError("mixed success/failure training data is incomplete")

    RUN.mkdir(parents=True)
    (RUN / "audit/baseline").mkdir(parents=True)
    REG.mkdir(parents=True)
    trainer = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
    prereg = {
        "format": "strict-track2-v208-mixed-right-gripper-contrast-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "teach the right-arm world-model expert the long-horizon visual and reward "
            "difference between public successful right-gripper closure and public on-policy "
            "open-gripper failure, so the unchanged official RL algorithm receives an action-dependent signal"
        ),
        "diagnosis": {
            "path": str(DIAGNOSIS),
            "sha256": sha256(DIAGNOSIS),
            "finding": "v207 moves right-arm joints but leaves action[13] open while action[6] closes",
        },
        "parent": str(PARENT),
        "parent_model_sha256": sha256(PARENT / "model.pt"),
        "trainer": str(trainer),
        "trainer_sha256": sha256(trainer),
        "data": {
            "mixed_windows": str(MIXED),
            "mixed_split": str(MIXED_SPLIT),
            "mixed_split_sha256": sha256(MIXED_SPLIT),
            "public_success_holdout_split": str(PUBLIC_SPLIT),
            "public_success_holdout_split_sha256": sha256(PUBLIC_SPLIT),
            "linked_public_success_windows": mixed["linked_official_success_windows"],
            "linked_public_onpolicy_windows": mixed["linked_onpolicy_train_windows"],
            "real_development_or_acceptance_data": False,
        },
        "fixed_training": {
            "steps": 1200,
            "checkpoint_steps": [200, 400, 600, 800, 1000, 1200],
            "batch_size": 1,
            "chunks": 4,
            "frames_per_sequence": 32,
            "arm_filter": "right",
            "learning_rate": 1e-7,
            "reward_objective": "probability",
            "reward_probability_scale_floor": 0.01,
            "reward_loss_weight": 0.1,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 10.0,
            "success_weight": 1.25,
            "late_start": 64,
            "late_weight": 4.0,
            "terminal_visual_weight": 2.0,
            "max_grad_norm": 1.0,
            "seed": 1407,
        },
        "fixed_audit": {
            "baseline": "v205 selected right expert with unchanged v202 left expert",
            "success_holdout": "untouched public 10-episode split, including four right-arm successful episodes",
            "failure_holdout": "pre-existing episode-disjoint public on-policy validation split",
            "recursive_chunks": 4,
            "candidate_order": [200, 400, 600, 800, 1000, 1200],
            "selection": "lowest mean of public-success and on-policy-failure right visual/reward/terminal ratios",
            "gates": {
                "public_right_visual_mae_ratio_max": 1.02,
                "public_right_reward_mae_ratio_max": 1.0,
                "public_right_terminal_gain_mae_ratio_max": 1.0,
                "failure_right_visual_mae_ratio_max": 1.02,
                "failure_right_reward_mae_ratio_max": 1.0,
                "failure_right_terminal_gain_mae_ratio_max": 1.05,
                "mean_ratio_must_improve": True,
            },
        },
        "guards": {
            "world_model_training_only": True,
            "fixed_official_initial_policy_preserved": True,
            "official_reward_model_frozen": True,
            "policy_behavior_cloning": False,
            "policy_action_injection": False,
            "participant_action_selection": False,
            "real_submission": False,
            "final_128_used_for_training_or_selection": False,
        },
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"run": str(RUN), "registry": str(REG), "preregistration_sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
