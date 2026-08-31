#!/usr/bin/env python3
"""Freeze the episode-disjoint public on-policy failure holdout for v208."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
SOURCE = JOINT / "onpolicy_windows_full128_stride4/split_manifest.json"
TRAINING = JOINT / "v163_mixed_reward_windows/split_manifest.json"
OUTPUT = JOINT / (
    "v208_v205_mixed_right_gripper_contrast_long32_seed1407/"
    "audit/failure_holdout_split.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit("refusing to overwrite frozen v208 failure holdout")
    source = json.loads(SOURCE.read_text())
    training = json.loads(TRAINING.read_text())
    validation = set(source["validation_episodes"])
    rows = [row for row in source["episodes"] if row["episode_id"] in validation]
    failures = sorted(row["episode_id"] for row in rows if not row["capture_success"])
    right_failures = sorted(
        row["episode_id"]
        for row in rows
        if not row["capture_success"] and row["arm"] == "right"
    )
    if len(validation) != 16 or len(failures) != 10 or len(right_failures) != 7:
        raise ValueError("unexpected public on-policy validation composition")
    overlap = sorted(set(training["train_episodes"]) & set(failures))
    if overlap:
        raise ValueError(f"failure holdout overlaps v208 training episodes: {overlap}")
    payload = {
        "format": "strict-track2-v208-public-onpolicy-failure-holdout-v1",
        "validation_episodes": failures,
        "failure_episode_count": len(failures),
        "right_failure_episodes": right_failures,
        "right_failure_episode_count": len(right_failures),
        "capture_success_required": False,
        "episode_disjoint_from_training": True,
        "training_episode_overlap": overlap,
        "source_split_manifest": str(SOURCE),
        "source_split_manifest_sha256": sha256(SOURCE),
        "training_split_manifest": str(TRAINING),
        "training_split_manifest_sha256": sha256(TRAINING),
        "hidden_or_final_evaluation_data": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
