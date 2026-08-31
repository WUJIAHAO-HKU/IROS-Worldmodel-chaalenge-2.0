#!/usr/bin/env python3
"""Pre-register the first full-horizon conservative RL run on v202."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFFICIAL = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v203_v202_conservativekl_h200_r2_step8_seed1403_20260818"
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
        raise SystemExit("refusing to overwrite v203 registration or run")
    REG.mkdir(parents=True)
    parent = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/selected_model/model.pt"
    policy = BASE / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    runner = BASE / "pipeline/scripts/run_strict_track2_conservative_kl.sh"
    p2_report = JOINT / "v202_v201_public_terminal_reward_calibration_seed1402/audit/p2_reward_calibration_report.json"
    prereg = {
        "format": "strict-track2-v203-v202-conservative-kl-preregistration-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Eight conservative full-horizon GRPO updates from the frozen official Pi0.5 policy using the public-only v202 parent world model.",
        "run_path": str(RUN),
        "frozen_training": {
            "max_steps": 8,
            "episode_steps": 200,
            "rollout_steps": 200,
            "rollout_epoch": 2,
            "total_envs": 8,
            "group_size": 4,
            "actor_global_batch_size": 1600,
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
            "p2_report": str(p2_report),
            "p2_report_sha256": sha256(p2_report),
        },
        "policy_initialization": {
            "source": "frozen official Pi0.5 baseline",
            "checkpoint": str(policy),
            "sha256": sha256(policy),
            "prior_rl_checkpoint_used": False,
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
    print(json.dumps({"registry": str(REG), "run": str(RUN), "world_model_sha256": prereg["world_model"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
