#!/usr/bin/env python3
"""Pre-register v203b after v203's zero-update batch divisibility failure."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v203b_v202_conservativekl_h200_r2_step8_seed1403_20260818"
REG = OFFICIAL / "run_registry" / NAME
RUN = OFFICIAL / "runs" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise SystemExit("refusing to overwrite v203b registration or run")
    REG.mkdir(parents=True)
    parent = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model/model.pt"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    failed_run = OFFICIAL / "runs/v203_v202_conservativekl_h200_r2_step8_seed1403_20260818"
    failure = failed_run / "audit/zero_update_failure.json"
    prereg = {
        "format": "strict-track2-v203b-v202-conservative-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor": {
            "run": str(failed_run),
            "status": "zero_policy_updates_configuration_failure",
            "failure_record": str(failure),
            "failure_record_sha256": sha256(failure),
            "finding": "rollout_size=400 must be divisible by actor global batch; inherited r8 value 1600 was invalid for r2",
        },
        "purpose": "Retry the identical conservative full-horizon run with the mechanically required actor global batch size.",
        "run_path": str(RUN),
        "frozen_training": {
            "max_steps": 8,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 400,
            "actor_lr": 2e-6,
            "kl_beta": 0.1,
            "kl_penalty": "low_var_kl",
            "actor_seed": 1403,
            "env_seed": 0,
            "reference_state_storage": "disk_mmap_during_kl_only",
            "terminal_checkpoint_contents": "full_model_weights_only",
        },
        "acceptance_gates": {
            "action_dim_normalized_approx_kl_abs_max": 0.05,
            "action_dim_normalized_clip_fraction_max": 0.15,
            "gradient_norm_max": 5.0,
            "all_update_kl_loss_finite_nonnegative": True,
            "checkpoint_zip_integrity_required": True,
        },
        "world_model": {
            "model_version": "track2-v202-public-terminal-calibrated-step75",
            "checkpoint": str(parent),
            "sha256": sha256(parent),
        },
        "runner": {"path": str(runner), "sha256": sha256(runner)},
        "post_training": {
            "selection_before_final_128": "public local simulator development split only",
            "real_submission": False,
            "final_128_evaluation_count_before_freeze": 0,
        },
        "prohibited_inputs": [
            "hidden development-evaluation outcomes",
            "final-evaluation seeds or outcomes",
            "real contest submission feedback",
        ],
        "claim_boundary": "Training and infrastructure candidate only; not a policy success-rate claim.",
    }
    (REG / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
