#!/usr/bin/env python3
"""Preregister a public-data-only one-update action-exploration diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818"
REG = OFF / "run_registry" / NAME
RUN = OFF / "runs" / NAME
RELEASE = JOINT / "v209_v202_v208_public_arm_routed_release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v211 diagnostic registration or run")
    REG.mkdir(parents=True)
    manifest = RELEASE / "arm_routed_autoregressive_manifest.json"
    policy = OFF.parent / "official_resources/pi05_adjust_bottle/model.safetensors"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    restart = BASE / "pipeline/scripts/restart_v209_services.sh"
    payload = {
        "format": "strict-track2-v211-train-action-exploration-diagnostic-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "measure whether the unchanged official training policy samples a physical "
            "right-gripper close action during public v209 world-model rollouts"
        ),
        "run": str(RUN),
        "world_model": {
            "release": str(RELEASE),
            "model_version": "track2-v209-public-arm-routed-v202-left-v208-right-selected",
            "manifest_sha256": sha256(manifest),
        },
        "frozen_training": {
            "initial_policy": "official unmodified Pi0.5 adjust_bottle checkpoint",
            "initial_policy_sha256": sha256(policy),
            "max_steps": 1,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 400,
            "actor_lr": 5e-6,
            "kl_beta": 0.05,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1410,
            "env_seed": 0,
            "openpi_noise_level": 0.3,
        },
        "capture": {
            # Keep capture artifacts in the preregistration tree: the RL launcher
            # deliberately refuses to start when its run directory already exists.
            "bridge_audit_dir": str(REG / "bridge_audit"),
            "max_items": 50,
            "expected_items": 50,
            "expected_future_action_shape": [8, 8, 14],
            "right_gripper_close_threshold": 0.5,
            "selection_use": False,
        },
        "source_hashes": {
            "runner": sha256(runner),
            "restart_v209_services": sha256(restart),
        },
        "guards": {
            "public_world_model_resets_only": True,
            "policy_action_injection": False,
            "policy_wrapper_change": False,
            "algorithm_change": False,
            "public_policy_evaluation": False,
            "final_128_evaluation": False,
            "real_submission": False,
            "hidden_or_official_request_data": False,
        },
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
