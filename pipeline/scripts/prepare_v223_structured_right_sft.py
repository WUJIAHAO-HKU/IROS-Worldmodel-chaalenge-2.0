#!/usr/bin/env python3
"""Preregister the public-data-only v223 structured right-arm run."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    output, base_ckpt, dataset, worker, model, launcher, smoke_report = map(
        pathlib.Path, sys.argv[1:]
    )
    smoke = json.loads(smoke_report.read_text())
    assert smoke["accepted"] is True
    assert smoke["metrics"]["sft_arm"] == 1.0
    assert smoke["metrics"]["sft_physical_action_weight_sum"] == 10.75
    info = json.loads((dataset / "meta" / "info.json").read_text())
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "v223_v169_structured_rightsft_kl_h8_step24_seed1423",
        "rules": {
            "real_submission": False,
            "hidden_or_reserved_evaluation_access": False,
            "training_data": "official/public successful demonstrations only",
            "selection_signal": "v222 one-update gradient contract; no policy outcome evaluation",
        },
        "rationale": {
            "root_cause": "v221 optimized all 32 action dimensions, including 18 padding dimensions",
            "data_audit": "transformed right-arm supervision has 1.60x last7/first7 mean absolute magnitude",
            "v222_smoke": smoke["metrics"],
            "change": "optimize only 14 physical dimensions with explicit arm1 and stronger right gripper weight",
        },
        "initial_policy": {
            "name": "v169_parent_step5_seed1243",
            "historical_public_score": "51/128",
            "path": str(base_ckpt),
            "sha256": digest(base_ckpt),
        },
        "sft": {
            "dataset": str(dataset),
            "source": "public clean50 train40 right-arm subset",
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "batch_size": 2,
            "num_workers": 0,
            "loss_weight": 0.1,
            "single_active_arm": 1,
            "structured_action_loss": True,
            "active_joint_weight": 1.0,
            "active_gripper_weight": 3.0,
            "inactive_keep_weight": 0.25,
            "physical_weight_sum": 10.75,
            "padded_weight": 0.0,
            "estimated_supervised_examples": 24 * 8 * 2,
        },
        "rl": {
            "parent_world_model": "v218 route-aware public model",
            "max_updates": 24,
            "episode_steps": 8,
            "rollout_epochs": 1,
            "group_size": 4,
            "total_envs": 8,
            "actor_lr": 1e-5,
            "kl_beta_to_initial_v169": 0.1,
        },
        "training_acceptance": {
            "valid_terminal_checkpoint": True,
            "all_24_updates_have_structured_sft_metrics": True,
            "every_sft_arm_is_1": True,
            "every_physical_weight_sum_is_10.75": True,
            "finite_positive_sft_loss_and_grad_norm": True,
            "no_oom_or_uncaught_exception": True,
        },
        "next_gate_if_training_passes": {
            "public_batch": "batch_00 (4 left, 12 right)",
            "thresholds": {
                "right_success": 1,
                "left_success": 2,
                "total_success": 3,
                "grasp_once": 12,
            },
            "on_reject": "do not evaluate public batches 01-06",
        },
        "source_hashes": {
            "fsdp_actor_worker.py": digest(worker),
            "openpi_action_model.py": digest(model),
            "run_strict_track2_conservative_kl.sh": digest(launcher),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("V223_STRUCTURED_RIGHT_SFT_PREREGISTERED")


if __name__ == "__main__":
    main()
