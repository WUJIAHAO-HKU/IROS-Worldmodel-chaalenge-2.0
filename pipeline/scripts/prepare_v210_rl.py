#!/usr/bin/env python3
"""Preregister the single-variable v210 RL test against accepted v209."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v210_v209_conservativekl_h200_r2_step8_lr5e6_beta005_seed1408_20260818"
REG = OFF / "run_registry" / NAME
RUN = OFF / "runs" / NAME
RELEASE = JOINT / "v209_v202_v208_public_arm_routed_release"
V208_AUDIT = JOINT / (
    "v208_v205_mixed_right_gripper_contrast_long32_seed1407/"
    "audit/v208_right_contrast_report.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v210 registration or run")
    audit = json.loads(V208_AUDIT.read_text())
    if audit.get("passed") is not True:
        raise ValueError("v208 two-holdout audit did not pass")
    REG.mkdir(parents=True)
    manifest = RELEASE / "arm_routed_autoregressive_manifest.json"
    release_doc = json.loads(manifest.read_text())
    left = RELEASE / "left_expert/model.pt"
    right = RELEASE / "right_expert/model.pt"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    payload = {
        "format": "strict-track2-v210-v209-conservative-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "isolate the effect of the accepted mixed success/failure right-arm world model "
            "while preserving the already-audited v207 policy-training configuration"
        ),
        "run_path": str(RUN),
        "world_model": {
            "model_version": release_doc["model_version"],
            "selected_v208_step": release_doc["selected_v208_step"],
            "release": str(RELEASE),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "left_model_sha256": sha256(left),
            "right_model_sha256": sha256(right),
            "public_two_holdout_audit": str(V208_AUDIT),
            "public_two_holdout_audit_sha256": sha256(V208_AUDIT),
        },
        "change_from_v207": {
            "only_intended_algorithmic_change": "v206 right expert -> accepted v208 right expert",
            "actor_seed": "new deterministic seed 1408 for a new run",
            "unchanged": [
                "fresh official Pi0.5 initialization",
                "8 updates and 2x200 rollout",
                "actor learning rate 5e-6",
                "KL beta 0.05 and action-normalized low_var_kl",
                "group size 4 and total environments 8",
            ],
        },
        "frozen_training": {
            "initial_policy": "official unmodified Pi0.5 adjust_bottle checkpoint",
            "max_steps": 8,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 400,
            "actor_lr": 5e-6,
            "kl_beta": 0.05,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1408,
            "env_seed": 0,
            "reference_state_storage": "disk_mmap_during_kl_only",
            "terminal_checkpoint_contents": "full_model_weights_only",
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.01,
            "action_dim_normalized_clip_fraction_max": 0.05,
            "gradient_norm_max": 5.0,
            "all_update_kl_loss_finite_nonnegative": True,
            "checkpoint_zip_integrity_required": True,
        },
        "public_policy_screen": {
            "stage_a": "19/32, +2 over baseline, both arms >=50%, grasps not below baseline",
            "stage_b": "75/112, +3 over baseline, both arms >=60%",
            "selection_data": "frozen public unseen train seeds only",
        },
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "guards": {
            "real_submission": False,
            "final_128_used_for_training_or_selection": False,
            "fixed_official_initial_policy": True,
            "policy_action_injection": False,
        },
        "prohibited_inputs": [
            "hidden development outcomes",
            "official final outcomes",
            "real contest submission feedback",
        ],
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
