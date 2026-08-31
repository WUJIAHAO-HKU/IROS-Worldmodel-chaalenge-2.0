#!/usr/bin/env python3
"""Preregister conservative RL against the accepted v206 routed world model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v207_v206_conservativekl_h200_r2_step8_lr5e6_beta005_seed1406_20260818"
REG = OFF / "run_registry" / NAME
RUN = OFF / "runs" / NAME
RELEASE = JOINT / "v206_v202_v205_public_arm_routed_release"
V205_AUDIT = JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/audit/right_terminal_parent_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v207 registration or run")
    audit = json.loads(V205_AUDIT.read_text())
    if audit.get("passed") is not True:
        raise ValueError("v205 public holdout audit did not pass")
    REG.mkdir(parents=True)
    manifest = RELEASE / "arm_routed_autoregressive_manifest.json"
    left = RELEASE / "left_expert/model.pt"
    right = RELEASE / "right_expert/model.pt"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    prereg = {
        "format": "strict-track2-v207-v206-conservative-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "test whether the public-holdout-accepted right-terminal world model supplies "
            "a usable right-arm policy gradient while preserving the official fixed start"
        ),
        "run_path": str(RUN),
        "world_model": {
            "model_version": "track2-v206-public-arm-routed-v202-left-v205-right-step120",
            "release": str(RELEASE),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "left_model_sha256": sha256(left),
            "right_model_sha256": sha256(right),
            "public_holdout_audit": str(V205_AUDIT),
            "public_holdout_audit_sha256": sha256(V205_AUDIT),
        },
        "change_from_prior_policy_runs": {
            "world_model": "v202 single parent -> v206 v202-left/v205-right routed parent",
            "actor_lr": "midpoint 5e-6 between unchanged v203b (2e-6) and grasp-regressing v204 (1e-5)",
            "kl_beta": 0.05,
            "unchanged": [
                "fresh official Pi0.5 initialization",
                "8 updates",
                "2x200 rollout",
                "group size 4",
                "true frozen-reference action-normalized low_var_kl",
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
            "actor_seed": 1406,
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
        },
        "prohibited_inputs": [
            "hidden development outcomes",
            "official final outcomes",
            "real contest submission feedback",
        ],
    }
    path = REG / "preregistration.json"
    path.write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
