#!/usr/bin/env python3
"""Preregister v226 before training on the public right-arm terminal subset."""

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
    (
        output,
        base_ckpt,
        dataset,
        dataset_audit_path,
        loader_audit_path,
        worker,
        model,
        launcher,
    ) = map(pathlib.Path, sys.argv[1:])
    dataset_audit = json.loads(dataset_audit_path.read_text())
    loader_audit = json.loads(loader_audit_path.read_text())
    assert dataset_audit["accepted"] is True
    assert dataset_audit["source_public_only"] is True
    assert dataset_audit["episodes"] == 15
    assert dataset_audit["frames"] == 1292
    assert loader_audit["accepted"] is True
    assert loader_audit["right_gripper_closed_fraction"] > 0.99
    assert loader_audit["padding_14_plus_max_abs"] == 0.0
    info = json.loads((dataset / "meta" / "info.json").read_text())
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "v226_v223_terminal_rightsft_kl_h8_step64_seed1426",
        "rules": {
            "real_submission": False,
            "hidden_or_reserved_evaluation_access": False,
            "training_data": "official/public successful demonstrations only",
            "selection_signal": "public batch00 diagnostic capture and public demonstration audit",
        },
        "rationale": {
            "diagnosis": (
                "v223 reached bottle contact in 10/12 right episodes, but only 5/12 ever "
                "commanded right-gripper closure and its post-close right-joint travel was "
                "0.955 versus 2.849 in successful public demonstrations"
            ),
            "change": (
                "continue v223 with causal examples from eight frames before closure through "
                "the terminal lift/place phase; retain structured physical-only right-arm loss"
            ),
            "loader_audit": loader_audit,
        },
        "initial_policy": {
            "name": "v223_v169_structured_rightsft_kl_h8_step24_seed1423",
            "path": str(base_ckpt),
            "sha256": digest(base_ckpt),
        },
        "sft": {
            "dataset": str(dataset),
            "source": "15 public right-arm successful demonstrations; terminal windows only",
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "batch_size": 2,
            "num_workers": 0,
            "loss_weight": 0.3,
            "single_active_arm": 1,
            "structured_action_loss": True,
            "active_joint_weight": 1.0,
            "active_gripper_weight": 3.0,
            "inactive_keep_weight": 0.25,
            "physical_weight_sum": 10.75,
            "padded_weight": 0.0,
            "estimated_supervised_examples": 64 * 8 * 2,
            "estimated_dataset_epochs": (64 * 8 * 2) / 1292,
        },
        "rl": {
            "parent_world_model": "v218 route-aware public model",
            "max_updates": 64,
            "episode_steps": 8,
            "rollout_epochs": 1,
            "group_size": 4,
            "total_envs": 8,
            "actor_lr": 1e-5,
            "kl_beta_to_initial_v223": 0.1,
        },
        "training_acceptance": {
            "valid_terminal_checkpoint": True,
            "all_64_updates_have_structured_sft_metrics": True,
            "every_sft_arm_is_1": True,
            "every_physical_weight_sum_is_10.75": True,
            "finite_positive_sft_loss_and_grad_norm": True,
            "no_oom_or_uncaught_exception": True,
        },
        "next_gate_if_training_passes": {
            "public_batch": "batch_00 (4 left, 12 right)",
            "thresholds": {
                "right_success": 2,
                "left_success": 2,
                "total_success": 4,
                "right_grasp_once": 10,
            },
            "on_reject": "do not evaluate public batches 01-06",
        },
        "source_hashes": {
            "fsdp_actor_worker.py": digest(worker),
            "openpi_action_model.py": digest(model),
            "run_strict_track2_conservative_kl.sh": digest(launcher),
            "dataset_audit.json": digest(dataset_audit_path),
            "transformed_loader_audit.json": digest(loader_audit_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("V226_TERMINAL_RIGHT_SFT_PREREGISTERED")


if __name__ == "__main__":
    main()
