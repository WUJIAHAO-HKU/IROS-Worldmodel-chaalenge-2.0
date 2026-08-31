#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
REGISTRY = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v426_v423_motion_guarded_right_release"
RELEASE = JOINT / "v426_v423_motion_guarded_right_release"
LEFT = JOINT / "v209_v202_v208_public_arm_routed_release/left_expert"
FROZEN_RIGHT = JOINT / "v209_v202_v208_public_arm_routed_release/right_expert"
LEARNED_RIGHT = JOINT / "v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823/selected_model"
RUNTIME = ROOT / "pipeline/wam_pipeline/v426_motion_guarded_right_runtime.py"
THRESHOLD = 0.03


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if RELEASE.exists() or REGISTRY.exists():
        raise FileExistsError("v426 release already exists")
    RELEASE.mkdir(parents=True)
    REGISTRY.mkdir(parents=True)
    os.symlink(LEFT.resolve(), RELEASE / "left_expert", target_is_directory=True)
    os.symlink(LEARNED_RIGHT.resolve(), RELEASE / "learned_right_expert", target_is_directory=True)
    os.symlink(FROZEN_RIGHT.resolve(), RELEASE / "frozen_right_expert", target_is_directory=True)
    manifest = {
        "format": "track2-v426-motion-guarded-right-release-v1",
        "model_version": "track2-v426-v423-low-motion-v208-high-motion-right",
        "left_expert": "left_expert",
        "learned_right_expert": "learned_right_expert",
        "frozen_right_expert": "frozen_right_expert",
        "right_normalized_motion_max": THRESHOLD,
        "right_motion": "mean abs joint delta / learned-right std over right joints 7:13; gripper excluded",
        "routing_inputs": ["instruction", "history_actions", "future_actions"],
        "prohibited_inputs": ["context_frames", "seed", "request_id", "request_order", "reward", "success", "evaluation result"],
        "does_not_score_or_modify_actions": True,
        "model_sha256": {
            "left": sha256(LEFT / "model.pt"),
            "learned_right": sha256(LEARNED_RIGHT / "model.pt"),
            "frozen_right": sha256(FROZEN_RIGHT / "model.pt"),
        },
        "runtime": str(RUNTIME),
        "runtime_sha256": sha256(RUNTIME),
        "selection_basis": "fixed v423 training high-motion boundary 0.03; no threshold sweep is deployed",
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    manifest_path = RELEASE / "motion_guarded_right_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    registration = {
        "format": "strict-track2-v426-motion-guarded-right-registration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release": str(RELEASE),
        "manifest_sha256": sha256(manifest_path),
        "authorization": "recursive public-world-model gate only; no RL or submission",
        "guards": {"outcomes_or_rewards_used_for_routing": False, "hidden_or_final_data": False, "real_submission": False},
    }
    (REGISTRY / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps(registration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
