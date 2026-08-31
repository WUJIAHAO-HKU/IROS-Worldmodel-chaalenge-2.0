#!/usr/bin/env python3
"""Package the accepted public-only right expert with the unchanged v202 left expert."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
TAG = "v206_v202_v205_public_arm_routed_release"
RELEASE = JOINT / TAG
REGISTRY = OFFICIAL / "run_registry" / TAG
LEFT = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
V205 = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405"
RIGHT = V205 / "selected_right_expert"
AUDIT = V205 / "audit/right_terminal_parent_report.json"


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
    if not audit.get("passed") or audit.get("selected", {}).get("step") != 120:
        raise ValueError("v205 did not pass its preregistered public holdout gates")
    if sha256(RIGHT / "model.pt") != audit["selected_model_sha256"]:
        raise ValueError("v205 selected expert hash no longer matches its audit")

    RELEASE.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    os.symlink(LEFT.resolve(), RELEASE / "left_expert", target_is_directory=True)
    os.symlink(RIGHT.resolve(), RELEASE / "right_expert", target_is_directory=True)
    os.symlink(AUDIT.resolve(), RELEASE / "right_expert_audit.json")
    manifest = {
        "format": "track2-arm-routed-autoregressive-release-v1",
        "model_version": "track2-v206-public-arm-routed-v202-left-v205-right-step120",
        "left_expert": "left_expert",
        "right_expert": "right_expert",
        "model_sha256": {
            "left_expert": sha256(LEFT / "model.pt"),
            "right_expert": sha256(RIGHT / "model.pt"),
        },
        "routing": {
            "classification": "fixed world-model expert routing only",
            "rule": (
                "explicit 'right arm'/'left arm' instruction first; otherwise choose the "
                "14-D action half with greater mean temporal-delta magnitude"
            ),
            "inputs": ["instruction", "history_actions", "future_actions"],
            "does_not_score_or_modify_actions": True,
            "prohibited_inputs": [
                "seed", "request_id", "request_order", "reward", "success", "evaluation result"
            ],
        },
        "right_expert_audit": "right_expert_audit.json",
        "right_expert_audit_sha256": sha256(AUDIT),
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    manifest_path = RELEASE / "arm_routed_autoregressive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prereg = {
        "format": "strict-track2-v206-arm-routed-release-registration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "release": str(RELEASE),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "model_version": manifest["model_version"],
        "purpose": "public-development RL from the unchanged official Pi0.5 starting policy",
        "selection_source": str(AUDIT),
        "selection_source_sha256": sha256(AUDIT),
        "guards": {
            "fixed_official_initial_policy": True,
            "participant_action_selection": False,
            "real_submission": False,
            "final_128_used": False,
        },
    }
    prereg_path = REGISTRY / "registration.json"
    prereg_path.write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"release": str(RELEASE), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
