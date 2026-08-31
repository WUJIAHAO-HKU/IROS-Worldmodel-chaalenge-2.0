#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
TAG = "v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823"
REGISTRY = ROOT / "artifacts/strict_track2_official_20260810/run_registry" / TAG
SOURCE = Path("/dev/shm") / TAG / "model/best"
RELEASE = ROOT / "artifacts/strict_track2_joint_augmentation_20260810" / TAG
SELECTED = RELEASE / "selected_model"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if RELEASE.exists():
        raise FileExistsError(RELEASE)
    preregistration = json.loads((REGISTRY / "preregistration.json").read_text())
    manifest = json.loads((SOURCE / "training_manifest.json").read_text())
    history = manifest["validation"]
    if [int(row["step"]) for row in history] != [0, 25, 50, 75, 100]:
        raise RuntimeError("unexpected v423 validation schedule")
    if int(manifest.get("best_checkpoint_step", -1)) != 100:
        raise RuntimeError("the uniquely selected v423 checkpoint is not step 100")
    baseline = float(history[0]["selection_metric"])
    selected = float(history[-1]["selection_metric"])
    relative_improvement = (baseline - selected) / baseline
    if relative_improvement < float(preregistration["fixed_training"]["minimum_relative_improvement_by_step50"]):
        raise RuntimeError("v423 did not pass its pre-registered proxy gate")
    SELECTED.mkdir(parents=True)
    for name in (
        "model.pt",
        "action_normalization.npz",
        "track2_autoregressive_unet_config.npz",
        "training_manifest.json",
    ):
        shutil.copy2(SOURCE / name, SELECTED / name)
    payload = {
        "format": "strict-track2-v423-mirror-augmented-right-dynamics-selection-v1",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint_step": 100,
        "selected_model": str(SELECTED),
        "step0_selection_metric": baseline,
        "step100_selection_metric": selected,
        "relative_selection_improvement": relative_improvement,
        "step0_rollout_mae": float(history[0]["rollout_mae"]),
        "step100_rollout_mae": float(history[-1]["rollout_mae"]),
        "step0_high_motion_rollout_mae": float(history[0]["high_motion_rollout_mae"]),
        "step100_high_motion_rollout_mae": float(history[-1]["high_motion_rollout_mae"]),
        "sha256": {name: sha256(SELECTED / name) for name in (
            "model.pt", "action_normalization.npz", "track2_autoregressive_unet_config.npz", "training_manifest.json"
        )},
        "evidence_sha256": {
            "preregistration": sha256(REGISTRY / "preregistration.json"),
            "training_log": sha256(REGISTRY / "training.log"),
        },
        "guards": {
            "selection_uses_real_public_right_holdout_only": True,
            "outcomes_or_rewards_read": False,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
            "rl_authorized": False,
        },
    }
    (RELEASE / "selection_registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
