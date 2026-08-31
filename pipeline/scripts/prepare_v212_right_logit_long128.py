#!/usr/bin/env python3
"""Preregister a public-data-only long-horizon right reward recovery pilot."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v212_v208_right_logit_long128_seed1411"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME
V208 = JOINT / "v208_v205_mixed_right_gripper_contrast_long32_seed1407"
V211 = OFF / "runs/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818"
MIXED = JOINT / "v163_mixed_reward_windows"
SPLIT = V208 / "training_split_arm_prompt_fixed.json"


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
    if v208_report.get("passed") is not True or v208_report["selected"]["step"] != 1200:
        raise ValueError("the frozen v208 right parent is not accepted at step 1200")
    right_parent = V208 / "selected_right_expert"
    if sha256(right_parent / "model.pt") != v208_report["selected_model_sha256"]:
        raise ValueError("v208 selected right model hash mismatch")

    action_audit_path = V211 / "audit/training_action_exploration.json"
    reward_audit_path = V211 / "audit/action_reward_alignment.json"
    action_audit = json.loads(action_audit_path.read_text())
    reward_audit = json.loads(reward_audit_path.read_text())
    explicit = reward_audit["explicit_right_instruction"]
    if not action_audit["diagnosis"]["physical_right_close_sampled"]:
        raise ValueError("v211 did not demonstrate physical right-close exploration")
    if explicit["right_close_fraction"]["mean"] < 0.25:
        raise ValueError("v211 explicit-right close coverage is unexpectedly low")
    if explicit["terminal_reward"]["max"] >= 0.01:
        raise ValueError("v211 no longer demonstrates the preregistered reward-collapse condition")

    sys.path.insert(0, str(BASE / "pipeline/scripts"))
    from train_multichunk_reward_aligned_autoregressive_unet import MultiChunkWindows

    split = json.loads(SPLIT.read_text())
    dataset = MultiChunkWindows(
        MIXED,
        split["train_episodes"],
        chunks=16,
        chunk_stride=8,
        arm_filter="right",
        instruction_by_episode={
            int(episode): str(instruction)
            for episode, instruction in split.get("episode_to_instruction", {}).items()
        },
    )
    success_count = int(dataset.capture_success.sum())
    failure_count = int((~dataset.capture_success).sum())
    if min(success_count, failure_count) < 1:
        raise ValueError("long128 inventory must contain both public successes and failures")

    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    trainer = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
    prereg = {
        "format": "strict-track2-v212-right-logit-long128-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "recover an absolute, non-saturated frozen-reward signal for sampled right-arm "
            "closure over 128 recursive public frames while preserving visual fidelity"
        ),
        "diagnosis": {
            "action_audit": str(action_audit_path),
            "action_audit_sha256": sha256(action_audit_path),
            "reward_audit": str(reward_audit_path),
            "reward_audit_sha256": sha256(reward_audit_path),
            "explicit_right_close_fraction_mean": explicit["right_close_fraction"]["mean"],
            "explicit_right_terminal_reward_max": explicit["terminal_reward"]["max"],
            "finding": (
                "official training samples right gripper closure, but v209 assigns every "
                "explicit-right trajectory a near-zero terminal reward"
            ),
        },
        "parent": str(right_parent),
        "parent_model_sha256": sha256(right_parent / "model.pt"),
        "trainer": str(trainer),
        "trainer_sha256": sha256(trainer),
        "data": {
            "mixed_public_windows": str(MIXED),
            "split": str(SPLIT),
            "split_sha256": sha256(SPLIT),
            "long128_right_sequences": len(dataset),
            "long128_success_sequences": success_count,
            "long128_failure_sequences": failure_count,
            "episode_count": int(len(set(dataset.episode_id.tolist()))),
            "real_development_or_acceptance_data": False,
        },
        "fixed_training": {
            "steps": 200,
            "checkpoint_steps": [50, 100, 150, 200],
            "batch_size": 1,
            "chunks": 16,
            "frames_per_sequence": 128,
            "arm_filter": "right",
            "learning_rate": 5e-8,
            "reward_objective": "logit",
            "reward_loss_weight": 0.02,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 4.0,
            "success_weight": 3.0,
            "late_start": 64,
            "late_weight": 4.0,
            "terminal_visual_weight": 2.0,
            "max_grad_norm": 1.0,
            "seed": 1411,
        },
        "fixed_audit": {
            "success_holdout": "episode-disjoint public 10-episode split",
            "failure_holdout": "episode-disjoint public on-policy failure split",
            "recursive_chunks": 16,
            "candidate_order": [50, 100, 150, 200],
            "release_requires_absolute_reward_gate": True,
            "absolute_success_threshold": 0.9,
            "right_success_visual_mae_ratio_max": 1.03,
            "right_failure_visual_mae_ratio_max": 1.03,
            "right_failure_false_positive_rate_max": 0.0,
        },
        "guards": {
            "world_model_training_only": True,
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
    print(json.dumps({"run": str(RUN), "registry": str(REG), "inventory": prereg["data"]}, indent=2))


if __name__ == "__main__":
    main()
