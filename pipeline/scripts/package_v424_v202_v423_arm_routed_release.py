#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
TAG = "v424_v202_v423_mirror_augmented_arm_routed_release"
RELEASE = JOINT / TAG
REGISTRY = OFFICIAL / "run_registry" / TAG
LEFT = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
V423 = JOINT / "v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823"
RIGHT = V423 / "selected_model"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if RELEASE.exists() or REGISTRY.exists():
        raise FileExistsError(TAG)
    selection = json.loads((V423 / "selection_registration.json").read_text())
    if int(selection.get("selected_checkpoint_step", -1)) != 100:
        raise RuntimeError("v423 selection is not frozen at step100")
    RELEASE.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    os.symlink(LEFT.resolve(), RELEASE / "left_expert", target_is_directory=True)
    os.symlink(RIGHT.resolve(), RELEASE / "right_expert", target_is_directory=True)
    os.symlink((V423 / "selection_registration.json").resolve(), RELEASE / "right_training_registration.json")
    manifest = {
        "format": "track2-arm-routed-autoregressive-release-v1",
        "model_version": "track2-v424-v202-left-v423-mirror-augmented-right",
        "left_expert": "left_expert",
        "right_expert": "right_expert",
        "selected_v423_step": 100,
        "model_sha256": {"left_expert": sha256(LEFT / "model.pt"), "right_expert": sha256(RIGHT / "model.pt")},
        "routing": {
            "classification": "fixed world-model expert routing only",
            "rule": "explicit arm instruction first; otherwise action-half temporal-delta magnitude",
            "inputs": ["instruction", "history_actions", "future_actions"],
            "does_not_score_or_modify_actions": True,
            "prohibited_inputs": ["seed", "request_id", "reward", "success", "evaluation result"],
        },
        "right_training_registration": "right_training_registration.json",
        "right_training_registration_sha256": sha256(V423 / "selection_registration.json"),
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    manifest_path = RELEASE / "arm_routed_autoregressive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    registration = {
        "format": "strict-track2-v424-arm-routed-release-registration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "release": str(RELEASE),
        "manifest_sha256": sha256(manifest_path),
        "model_version": manifest["model_version"],
        "authorization": "recursive public world-model diagnosis only; no RL or submission",
        "guards": {"participant_action_selection": False, "real_submission": False, "final_128_used": False},
    }
    (REGISTRY / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(registration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
