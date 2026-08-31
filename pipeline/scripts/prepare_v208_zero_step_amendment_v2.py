#!/usr/bin/env python3
"""Freeze the final zero-update v208 prompt-namespace correction."""

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
ORIGINAL = JOINT / "v163_mixed_reward_windows/split_manifest.json"
V205_SPLIT = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
TRAIN_SPLIT = RUN / "training_split_arm_prompt_fixed.json"
AMENDMENT_V1 = REG / "zero_step_training_amendment.json"
OUTPUT = REG / "zero_step_training_amendment_v2.json"
TRAIN_SCRIPT = BASE / "pipeline/scripts/run_v208_training.sh"
TRAINER = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
PREREG = REG / "preregistration.json"
EXPECTED_PREREG = "48e22442f55c5fb8c1d50a4c41c9cdb6ea5410212915915fc3a7e7c06e3b6016"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or TRAIN_SPLIT.exists():
        raise SystemExit("refusing to overwrite v208 amendment-v2 artifacts")
    if sha256(PREREG) != EXPECTED_PREREG or not AMENDMENT_V1.is_file():
        raise ValueError("v208 original preregistration or amendment-v1 is missing")
    if (RUN / "checkpoints").exists() or (RUN / "TRAINING_COMPLETE").exists():
        raise ValueError("amendment-v2 is only valid before any checkpoint exists")
    failure_log = RUN / "training.log"
    text = failure_log.read_text(errors="replace")
    if "episode instruction is not explicitly arm-consistent" not in text:
        raise ValueError("second pre-update prompt failure is not present")

    source = json.loads(ORIGINAL.read_text())
    v205 = json.loads(V205_SPLIT.read_text())
    official_offset = int(source["official_offset"])
    official_train = sorted(
        episode for episode in source["train_episodes"] if episode >= official_offset
    )
    prompts = {
        str(episode): v205["episode_to_instruction"][str(episode - official_offset)]
        for episode in official_train
    }
    for episode, prompt in prompts.items():
        arm = v205["arm_by_episode"][str(int(episode) - official_offset)]
        expected = f"{arm} arm"
        opposite = "left arm" if arm == "right" else "right arm"
        lowered = prompt.lower()
        if expected not in lowered or opposite in lowered:
            raise ValueError(f"official prompt is not explicit and arm-correct: {episode}")
    if len(prompts) != 40:
        raise ValueError("expected 40 public official training prompts")

    corrected = dict(source)
    corrected.update({
        "format": "strict-track2-v208-mixed-reward-training-split-prompt-fixed-v1",
        "source_split_manifest": str(ORIGINAL),
        "source_split_manifest_sha256": sha256(ORIGINAL),
        "episode_to_instruction": prompts,
        "prompt_policy": (
            "official demonstrations use v205 audited explicit arm-correct episode prompts; "
            "on-policy episodes intentionally omit a mapping and use the trainer's frozen "
            "arm-matched official prompt pool"
        ),
        "train_episode_membership_unchanged": True,
        "validation_episode_membership_unchanged": True,
    })
    TRAIN_SPLIT.write_text(json.dumps(corrected, indent=2) + "\n")

    prereg = json.loads(PREREG.read_text())
    if sha256(TRAINER) != prereg["trainer_sha256"]:
        raise ValueError("frozen trainer changed")
    payload = {
        "format": "strict-track2-v208-zero-update-technical-amendment-v2",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "original_preregistration": str(PREREG),
        "original_preregistration_sha256": EXPECTED_PREREG,
        "prior_amendment": str(AMENDMENT_V1),
        "prior_amendment_sha256": sha256(AMENDMENT_V1),
        "second_failure_log": str(failure_log),
        "second_failure_log_sha256": sha256(failure_log),
        "zero_optimizer_updates": True,
        "zero_checkpoints_created": True,
        "cause": "mixed manifest carried non-explicit random descriptions for official demos",
        "correction": "replace only prompt mapping with the already-audited v205 explicit prompts",
        "corrected_training_split": str(TRAIN_SPLIT),
        "corrected_training_split_sha256": sha256(TRAIN_SPLIT),
        "source_training_split_sha256": sha256(ORIGINAL),
        "v205_prompt_split_sha256": sha256(V205_SPLIT),
        "corrected_training_script_sha256": sha256(TRAIN_SCRIPT),
        "trainer_sha256": sha256(TRAINER),
        "unchanged": [
            "all train and validation episode memberships",
            "all image/action tensors and capture outcomes",
            "parent, trainer, loss, hyperparameters, steps, seed, and candidate order",
            "success and failure holdout audits",
        ],
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
