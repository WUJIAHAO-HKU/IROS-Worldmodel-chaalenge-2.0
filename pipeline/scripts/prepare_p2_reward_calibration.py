#!/usr/bin/env python3
"""Freeze the public-only P2 reward-calibration protocol before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
RUN = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402"
REGISTRY = OFFICIAL / "run_registry/v202_v201_public_terminal_reward_calibration_seed1402"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REGISTRY.exists():
        raise SystemExit("refusing to overwrite an existing v202 registration")
    RUN.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)

    v192_path = JOINT / "v192_v177_official_demo_visual_mix/split_manifest.json"
    mixed_path = JOINT / "v163_mixed_reward_windows/split_manifest.json"
    onpolicy_path = JOINT / "onpolicy_windows_full128_stride4/split_manifest.json"
    parent = JOINT / "v201_v196_public_long16_sourceweighted_seed1401/model"
    v192 = json.loads(v192_path.read_text())
    mixed = json.loads(mixed_path.read_text())
    onpolicy = json.loads(onpolicy_path.read_text())

    train = [int(value) for value in v192["train_episodes"]]
    validation = [int(value) for value in v192["validation_episodes"]]
    if set(train) & set(validation):
        raise ValueError("v192 train and validation episodes overlap")
    if len(validation) != 8 or not all(value >= 20000 for value in validation):
        raise ValueError("expected the frozen eight public-demo validation episodes")
    episode_to_instruction = {
        str(episode): instruction
        for episode, instruction in mixed["episode_to_instruction"].items()
        if int(episode) in set(train) | set(validation)
    }
    missing = sorted(set(validation) - {int(value) for value in episode_to_instruction})
    if missing:
        raise ValueError(f"missing public-demo instructions: {missing}")

    split = {
        "format": "strict-track2-v202-public-reward-calibration-split-v1",
        "source_split": str(v192_path),
        "source_split_sha256": sha256(v192_path),
        "train_episodes": train,
        "validation_episodes": validation,
        "episode_to_instruction": episode_to_instruction,
        "validation_rule": v192["official_validation_rule"],
        "provenance": "declared public Track2 training demonstrations only",
        "hidden_or_final_evaluation_data": False,
    }
    split_path = RUN / "p2_split_manifest.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")

    preregistration = {
        "format": "strict-track2-v202-public-terminal-reward-calibration-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": "improve frozen-official-reward terminal/progress alignment while preserving the v201 visual parent",
        "parent": str(parent),
        "parent_model_sha256": sha256(parent / "model.pt"),
        "data": {
            "training_windows": str(JOINT / "v192_v177_official_demo_visual_mix/windows"),
            "training_split": str(split_path),
            "success_holdout": {
                "episodes": validation,
                "selection_independent_of_outcome": True,
                "source": "eight held-out declared public training demonstrations",
            },
            "onpolicy_holdout": {
                "windows": str(JOINT / "onpolicy_windows_full128_stride4"),
                "split": str(onpolicy_path),
                "episodes": [int(value) for value in onpolicy["validation_episodes"]],
                "source": "public RoboTwin simulator with frozen official Pi0.5",
            },
            "exact_instruction_map": str(JOINT / "reward_alignment/exact_instruction_map128.json"),
            "hidden_or_final_evaluation_data": False,
        },
        "fixed_training": {
            "steps": 100,
            "checkpoint_steps": [25, 50, 75, 100],
            "batch_size": 1,
            "learning_rate": 5e-8,
            "reward_objective": "probability",
            "reward_probability_scale_floor": 0.01,
            "reward_loss_weight": 0.1,
            "reward_delta_weight": 2.0,
            "reward_terminal_weight": 10.0,
            "right_weight": 2.0,
            "success_weight": 2.0,
            "late_weight": 3.0,
            "late_start": 80,
            "official_weight": 1.5,
            "high_motion_weight": 2.0,
            "high_motion_threshold": 0.03,
            "parent_anchor_weight": 0.1,
            "seed": 1402,
        },
        "fixed_audit": {
            "recursive_chunks": 2,
            "max_sequences_per_holdout": 96,
            "motion_threshold": 0.03,
            "strata": [
                "left_low_motion", "left_high_motion",
                "right_low_motion", "right_high_motion",
            ],
            "candidate_order": [0, 25, 50, 75, 100],
            "selection_metric": "mean relative terminal-gain-MAE plus reward-MAE over both holdouts",
            "gates": {
                "visual_mae_max_regression_percent": 1.0,
                "overall_reward_mae_max_regression_percent": 1.0,
                "overall_terminal_gain_mae_max_regression_percent": 1.0,
                "right_terminal_gain_mae_max_regression_percent": 1.0,
                "right_high_motion_reward_mae_max_regression_percent": 2.0,
                "must_improve_terminal_or_reward_metric": True,
            },
        },
        "guards": {
            "official_reward_model_frozen": True,
            "policy_actions_modified": False,
            "participant_action_selection": False,
            "mpc": False,
            "real_submission": False,
            "final_128_used_for_training_or_selection": False,
        },
    }
    prereg_path = REGISTRY / "preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2) + "\n")
    (RUN / "audit").mkdir()
    print(json.dumps({
        "run": str(RUN),
        "preregistration": str(prereg_path),
        "split": str(split_path),
        "train_episodes": len(train),
        "success_holdout_episodes": len(validation),
        "onpolicy_holdout_episodes": len(onpolicy["validation_episodes"]),
    }, indent=2))


if __name__ == "__main__":
    main()
