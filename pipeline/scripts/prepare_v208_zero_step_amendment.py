#!/usr/bin/env python3
"""Record the zero-update v208 instruction-namespace correction before retry."""

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
PREREG = REG / "preregistration.json"
OUTPUT = REG / "zero_step_training_amendment.json"
TRAIN_SCRIPT = BASE / "pipeline/scripts/run_v208_training.sh"
TRAINER = BASE / "pipeline/scripts/train_multichunk_reward_aligned_autoregressive_unet.py"
EXPECTED_PREREG = "48e22442f55c5fb8c1d50a4c41c9cdb6ea5410212915915fc3a7e7c06e3b6016"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite v208 zero-step amendment")
    if sha256(PREREG) != EXPECTED_PREREG:
        raise ValueError("original v208 preregistration hash changed")
    if (RUN / "checkpoints").exists() or (RUN / "TRAINING_COMPLETE").exists():
        raise ValueError("amendment is only valid before any checkpoint exists")
    log = RUN / "training.log"
    text = log.read_text(errors="replace")
    expected = "episode instruction is not explicitly arm-consistent"
    if expected not in text or 'optimizer.step' in text:
        raise ValueError("training log does not prove the expected pre-update failure")
    prereg = json.loads(PREREG.read_text())
    if sha256(TRAINER) != prereg["trainer_sha256"]:
        raise ValueError("frozen trainer changed")
    payload = {
        "format": "strict-track2-v208-zero-update-technical-amendment-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "original_preregistration": str(PREREG),
        "original_preregistration_sha256": EXPECTED_PREREG,
        "failure_log": str(log),
        "failure_log_sha256": sha256(log),
        "zero_optimizer_updates": True,
        "zero_checkpoints_created": True,
        "cause": (
            "official demonstration windows omit synthetic_seed; the optional on-policy "
            "instruction map therefore used its legacy default seed 0 before the correct "
            "episode-level official instruction fallback"
        ),
        "correction": (
            "omit the optional on-policy instruction map from training only; public official "
            "demonstrations keep their episode-level explicit arm-correct instructions and "
            "on-policy rows use the trainer's frozen arm-matched official prompt pool"
        ),
        "unchanged": [
            "training data and episode split",
            "parent checkpoint",
            "trainer source and loss",
            "all hyperparameters, steps, seed, and candidate order",
            "success and failure holdout audits",
        ],
        "corrected_training_script": str(TRAIN_SCRIPT),
        "corrected_training_script_sha256": sha256(TRAIN_SCRIPT),
        "trainer_sha256": sha256(TRAINER),
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
