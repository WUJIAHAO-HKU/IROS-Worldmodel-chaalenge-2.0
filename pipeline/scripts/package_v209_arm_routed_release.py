#!/usr/bin/env python3
"""Package the v208-selected right expert with the unchanged v202 left expert."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFF = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v209_v202_v208_public_arm_routed_release"
RELEASE = JOINT / TAG
REGISTRY = OFF / "run_registry" / TAG
LEFT = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
V208 = JOINT / "v208_v205_mixed_right_gripper_contrast_long32_seed1407"
RIGHT = V208 / "selected_right_expert"
AUDIT = V208 / "audit/v208_right_contrast_report.json"
FAILURE_SPLIT = V208 / "audit/failure_holdout_split.json"
MODEL_VERSION = "track2-v209-public-arm-routed-v202-left-v208-right-selected"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RELEASE.exists() or REGISTRY.exists():
        raise SystemExit(f"refusing to overwrite existing {TAG}")
    audit = json.loads(AUDIT.read_text())
    selected = audit.get("selected")
    if audit.get("passed") is not True or not selected:
        raise ValueError("v208 did not pass its preregistered two-holdout gates")
    if sha256(RIGHT / "model.pt") != audit["selected_model_sha256"]:
        raise ValueError("v208 selected expert hash no longer matches its audit")
    failure = json.loads(FAILURE_SPLIT.read_text())
    if failure.get("episode_disjoint_from_training") is not True:
        raise ValueError("v208 failure holdout is not episode-disjoint")

    RELEASE.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    os.symlink(LEFT.resolve(), RELEASE / "left_expert", target_is_directory=True)
    os.symlink(RIGHT.resolve(), RELEASE / "right_expert", target_is_directory=True)
    os.symlink(AUDIT.resolve(), RELEASE / "right_expert_audit.json")
    os.symlink(FAILURE_SPLIT.resolve(), RELEASE / "failure_holdout_split.json")
    manifest = {
        "format": "track2-arm-routed-autoregressive-release-v1",
        "model_version": MODEL_VERSION,
        "left_expert": "left_expert",
        "right_expert": "right_expert",
        "selected_v208_step": int(selected["step"]),
        "model_sha256": {
            "left_expert": sha256(LEFT / "model.pt"),
            "right_expert": sha256(RIGHT / "model.pt"),
        },
        "routing": {
            "classification": "fixed world-model expert routing only",
            "rule": "explicit arm instruction first; otherwise action-half temporal-delta magnitude",
            "inputs": ["instruction", "history_actions", "future_actions"],
            "does_not_score_or_modify_actions": True,
            "prohibited_inputs": ["seed", "request_id", "reward", "success", "evaluation result"],
        },
        "right_expert_audit": "right_expert_audit.json",
        "right_expert_audit_sha256": sha256(AUDIT),
        "failure_holdout_split": "failure_holdout_split.json",
        "failure_holdout_split_sha256": sha256(FAILURE_SPLIT),
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    manifest_path = RELEASE / "arm_routed_autoregressive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    registration = {
        "format": "strict-track2-v209-arm-routed-release-registration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "release": str(RELEASE),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "model_version": MODEL_VERSION,
        "selection_source": str(AUDIT),
        "selection_source_sha256": sha256(AUDIT),
        "guards": {
            "fixed_official_initial_policy": True,
            "participant_action_selection": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    path = REGISTRY / "registration.json"
    path.write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps({"release": str(RELEASE), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
