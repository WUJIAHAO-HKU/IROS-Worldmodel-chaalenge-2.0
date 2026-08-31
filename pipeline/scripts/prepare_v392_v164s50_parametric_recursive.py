#!/usr/bin/env python3
"""Freeze a single historical v164-step50 parametric-parent comparison."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v392_v164s50_vs_v209_parametric_recursive_seed1553_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    baseline = J / "v209_v202_v208_public_arm_routed_release"
    right = J / "v164_baseline_initialized_right_expert/checkpoint_step_000050"
    audit = ROOT / "pipeline/scripts/audit_v392_v164s50_parametric_recursive.py"
    old_screen = J / "v164_baseline_initialized_right_expert/screen_step50/reward.json"
    old_visual = J / "v164_baseline_initialized_right_expert/screen_step50/visual.json"
    old_prereg = J / "v164_baseline_initialized_right_expert/preregistration.json"
    for path in (
        baseline / "left_expert/model.pt",
        baseline / "right_expert/model.pt",
        right / "model.pt",
        right / "action_normalization.npz",
        right / "track2_autoregressive_unet_config.npz",
        audit,
        old_screen,
        old_visual,
        old_prereg,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Architecture and normalization must be exactly deployment-compatible.
    for name in ("action_normalization.npz", "track2_autoregressive_unet_config.npz"):
        if sha256(right / name) != sha256(baseline / "right_expert" / name):
            raise RuntimeError(f"v164/v209 compatibility mismatch: {name}")

    RUN.mkdir(parents=True)
    REG.mkdir(parents=True)
    release = RUN / "release"
    release.mkdir()
    os.symlink((baseline / "left_expert").resolve(), release / "left_expert", target_is_directory=True)
    os.symlink(right.resolve(), release / "right_expert", target_is_directory=True)
    manifest = {
        "format": "track2-arm-routed-autoregressive-release-v1",
        "model_version": "track2-v392-v202-left-v164-step50-right-diagnostic",
        "left_expert": "left_expert",
        "right_expert": "right_expert",
        "model_sha256": {
            "left_expert": sha256(baseline / "left_expert/model.pt"),
            "right_expert": sha256(right / "model.pt"),
        },
        "routing": {
            "classification": "fixed world-model expert routing only",
            "rule": "explicit arm instruction first; otherwise action-half temporal-delta magnitude",
            "inputs": ["instruction", "history_actions", "future_actions"],
            "does_not_score_or_modify_actions": True,
            "prohibited_inputs": ["seed", "request_id", "reward", "success", "evaluation result"],
        },
        "right_training": "historical preregistered v164 step50; public world-model data only",
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    manifest_path = release / "arm_routed_autoregressive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prereg = {
        "format": "strict-track2-v392-v164s50-parametric-recursive-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "historical v164 step50 reward-aligned right expert may improve v209 before retrieval/runtime attenuation",
        "fixed_candidate": manifest["model_version"],
        "coverage": "same frozen validation/local public right episodes; all 512 recursive windows",
        "fixed_checks": {
            "teacher_rgb_ratio_max_both": 1.00,
            "recursive_rgb_ratio_max_both": 1.00,
            "teacher_temporal_ratio_max_both": 1.02,
            "recursive_temporal_ratio_max_both": 1.02,
            "recursive_reward_error_ratio_max_both": 1.02,
            "all_required": True,
        },
        "decision": "failure permanently rejects v164-step50 reintegration; pass authorizes one full v390 integration audit only",
        "no_parameter_or_checkpoint_sweep": True,
        "evidence_sha256": {
            "audit": sha256(audit),
            "baseline_manifest": sha256(baseline / "arm_routed_autoregressive_manifest.json"),
            "candidate_manifest": sha256(manifest_path),
            "v164_preregistration": sha256(old_prereg),
            "v164_reward_screen": sha256(old_screen),
            "v164_visual_screen": sha256(old_visual),
        },
        "guards": {
            "public_world_model_data_only": True,
            "outcomes_used_at_runtime": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "release": str(release)}, indent=2))


if __name__ == "__main__":
    main()
