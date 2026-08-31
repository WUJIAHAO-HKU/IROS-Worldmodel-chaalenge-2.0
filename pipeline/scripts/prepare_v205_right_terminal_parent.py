#!/usr/bin/env python3
"""Preregister a public-only right-terminal Track 2 parent experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v205_v202_public_right_terminal_multichunk_seed1405"
RUN = JOINT / TAG
REGISTRY = OFFICIAL / "run_registry" / TAG
DATA = BASE / "artifacts/datasets/aloha-agilex_clean_50"
WINDOWS = BASE / "artifacts/adjust_bottle_windows_full"
RESET = BASE / "artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"
PARENT = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_arm(episode: int) -> str:
    path = DATA / "data" / f"episode{episode}.hdf5"
    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"], dtype=np.float32)
    delta = np.abs(np.diff(actions, axis=0))
    left = float(delta[:, :7].sum())
    right = float(delta[:, 7:].sum())
    if max(left, right) <= 0 or min(left, right) > 1e-6 * max(left, right):
        raise ValueError(
            f"episode {episode} does not have one unambiguous active arm: "
            f"left={left}, right={right}"
        )
    return "right" if right > left else "left"


def arm_correct_instruction(episode: int, arm: str) -> str:
    document = json.loads(
        (DATA / "instructions" / f"episode{episode}.json").read_text()
    )
    expected = f"{arm} arm"
    opposite = "left arm" if arm == "right" else "right arm"
    candidates = [
        str(value)
        for split in ("seen", "unseen")
        for value in document.get(split, [])
        if expected in str(value).lower() and opposite not in str(value).lower()
    ]
    if not candidates:
        raise ValueError(f"episode {episode} has no explicit {expected} instruction")
    return candidates[0]


def dataset_merkle(episodes: list[int]) -> str:
    digest = hashlib.sha256()
    for episode in sorted(episodes):
        path = DATA / "data" / f"episode{episode}.hdf5"
        digest.update(f"episode{episode}.hdf5\0{sha256(path)}\n".encode())
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REGISTRY.exists():
        raise SystemExit(f"refusing to overwrite existing {TAG}")
    if len(list((DATA / "data").glob("episode*.hdf5"))) != 50:
        raise ValueError("expected all 50 declared public demonstration episodes")

    reset = json.loads(RESET.read_text())
    train = sorted(int(value) for value in reset["episode_ids"])
    all_episodes = list(range(50))
    validation = sorted(set(all_episodes) - set(train))
    if len(train) != 40 or len(validation) != 10 or set(train) & set(validation):
        raise ValueError("expected the immutable public 40/10 episode split")

    arms = {episode: episode_arm(episode) for episode in all_episodes}
    prompts = {
        episode: arm_correct_instruction(episode, arms[episode])
        for episode in all_episodes
    }
    if sum(arms[value] == "right" for value in validation) != 4:
        raise ValueError("expected four untouched right-arm validation episodes")
    if sum(arms[value] == "right" for value in train) != 15:
        raise ValueError("expected fifteen right-arm training episodes")

    RUN.mkdir(parents=True)
    (RUN / "audit" / "baseline").mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    split = {
        "format": "strict-track2-v205-public-demo-40train-10holdout-v1",
        "source": "declared Track 2 aloha-agilex_clean_50 world-model fine-tuning data",
        "windows": str(WINDOWS),
        "train_episodes": train,
        "validation_episodes": validation,
        "arm_by_episode": {str(key): value for key, value in arms.items()},
        "episode_to_instruction": {
            str(key): value for key, value in prompts.items()
        },
        "instruction_rule": (
            "first seen, then unseen prompt that explicitly names the action-derived "
            "active arm and never names the opposite arm"
        ),
        "split_rule": (
            "the 40 public episodes in the pre-existing official reset manifest are "
            "training; its untouched 10-episode complement is validation"
        ),
        "train_arm_counts": {
            "left": sum(arms[value] == "left" for value in train),
            "right": sum(arms[value] == "right" for value in train),
        },
        "validation_arm_counts": {
            "left": sum(arms[value] == "left" for value in validation),
            "right": sum(arms[value] == "right" for value in validation),
        },
        "hidden_or_final_evaluation_data": False,
    }
    split_path = RUN / "public_demo_split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")

    trainer = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
    preregistration = {
        "format": "strict-track2-v205-public-right-terminal-parent-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "learn right-arm post-grasp and terminal dynamics from declared successful "
            "public demonstrations without changing the fixed Pi0.5 policy start"
        ),
        "parent": str(PARENT),
        "parent_model_sha256": sha256(PARENT / "model.pt"),
        "trainer": str(trainer),
        "trainer_sha256": sha256(trainer),
        "data": {
            "hdf5_root": str(DATA / "data"),
            "windows": str(WINDOWS),
            "split": str(split_path),
            "split_sha256": sha256(split_path),
            "public_50_hdf5_merkle_sha256": dataset_merkle(all_episodes),
            "train_episodes": train,
            "validation_episodes": validation,
            "train_right_episodes": [value for value in train if arms[value] == "right"],
            "validation_right_episodes": [
                value for value in validation if arms[value] == "right"
            ],
            "hidden_or_final_evaluation_data": False,
        },
        "fixed_training": {
            "steps": 120,
            "checkpoint_steps": [30, 60, 90, 120],
            "batch_size": 1,
            "chunks": 4,
            "frames_per_sequence": 32,
            "arm_filter": "right",
            "learning_rate": 1e-7,
            "reward_objective": "probability",
            "reward_probability_scale_floor": 0.01,
            "reward_loss_weight": 0.05,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 10.0,
            "late_start": 64,
            "late_weight": 6.0,
            "terminal_visual_weight": 2.0,
            "max_grad_norm": 1.0,
            "seed": 1405,
        },
        "fixed_audit": {
            "baseline": "v202 selected step75",
            "holdout": "the untouched public 10-episode complement, including four right-arm episodes",
            "recursive_chunks": 4,
            "candidate_order": [30, 60, 90, 120],
            "selection": (
                "lowest mean of right reward-MAE ratio and right terminal-gain-MAE ratio"
            ),
            "gates": {
                "right_visual_mae_max_regression_percent": 1.0,
                "right_reward_mae_ratio_max": 1.0,
                "right_terminal_gain_mae_ratio_max": 1.0,
                "right_high_motion_reward_mae_ratio_max": 1.02,
                "must_improve_right_selection_metric": True,
            },
        },
        "deployment": {
            "left_expert": "unchanged v202 selected step75",
            "right_expert": "the uniquely selected v205 checkpoint",
            "router_inputs": ["instruction", "history_actions", "future_actions"],
            "prohibited_router_inputs": [
                "seed", "request_id", "request_order", "reward", "success", "evaluation result"
            ],
        },
        "guards": {
            "fixed_official_initial_policy_preserved": True,
            "official_reward_model_frozen": True,
            "real_submission": False,
            "final_128_used_for_training_or_selection": False,
        },
    }
    prereg_path = REGISTRY / "preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tag": TAG,
                "run": str(RUN),
                "preregistration": str(prereg_path),
                "train": split["train_arm_counts"],
                "validation": split["validation_arm_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
