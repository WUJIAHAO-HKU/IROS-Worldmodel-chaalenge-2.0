#!/usr/bin/env python3
"""Pre-register v204 after v203b's public 32-seed no-effect rejection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v204_v202_effectivekl_h200_r2_step8_lr1e5_beta005_seed1404_20260818"
REG, RUN = OFF / "run_registry" / NAME, OFF / "runs" / NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v204 registration or run")
    REG.mkdir(parents=True)
    predecessor = OFF / "runs/v203b_v202_conservativekl_h200_r2_step8_seed1403_20260818"
    rejection = predecessor / "audit/public_policy_stage_a_acceptance.json"
    training_audit = predecessor / "audit/p3_training_acceptance.json"
    parent = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model/model.pt"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    rejected = json.loads(rejection.read_text(encoding="utf-8"))
    assert rejected["passed"] is False
    assert rejected["baseline"]["all"]["successes"] == 8
    assert rejected["candidate"]["all"]["successes"] == 8
    assert rejected["candidate"]["right"]["successes"] == 0
    prereg = {
        "format": "strict-track2-v204-v202-effective-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Increase effective policy movement after a numerically safe but behaviorally unchanged conservative run.",
        "predecessor": {
            "run": str(predecessor),
            "training_audit": str(training_audit),
            "training_audit_sha256": sha256(training_audit),
            "public_rejection": str(rejection),
            "public_rejection_sha256": sha256(rejection),
            "finding": "v203b matched baseline at 8/32 and 0/21 right-arm successes; grasp improved only 28 to 29",
        },
        "change_from_predecessor": {
            "actor_lr": {"from": 2e-6, "to": 1e-5},
            "kl_beta": {"from": 0.1, "to": 0.05},
            "unchanged": ["fresh official Pi0.5 initialization", "v202 world model", "8 updates", "2x200 rollout", "group size 4", "true frozen-reference low_var_kl"],
        },
        "run_path": str(RUN),
        "frozen_training": {
            "max_steps": 8,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 400,
            "actor_lr": 1e-5,
            "kl_beta": 0.05,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1404,
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
        "world_model": {"model_version": "track2-v202-public-terminal-calibrated-step75", "checkpoint": str(parent), "sha256": sha256(parent)},
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "post_training": {
            "selection_before_final_128": "same frozen public unseen train-seed development split only",
            "stage_a_gate_unchanged": "19/32, +2 over baseline, both arms >=50%",
            "real_submission": False,
            "final_128_evaluation_count_before_freeze": 0,
        },
        "prohibited_inputs": ["hidden development outcomes", "official final outcomes", "real contest submission feedback"],
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
