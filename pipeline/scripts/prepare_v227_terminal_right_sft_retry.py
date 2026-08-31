#!/usr/bin/env python3
"""Preregister the resource-only retry of v226."""

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
    path_args = list(map(pathlib.Path, sys.argv[1:9]))
    output, base_ckpt, dataset_audit_path, loader_audit_path, failure_path, device_path, rollout_worker, service_script = path_args
    candidate = sys.argv[9]
    expandable_segments = sys.argv[10].lower() == "true"
    initial_policy_name = sys.argv[11]
    actor_seed = int(sys.argv[12])
    actor_lr = float(sys.argv[13])
    active_joint_weight = float(sys.argv[14])
    active_gripper_weight = float(sys.argv[15])
    inactive_keep_weight = float(sys.argv[16])
    physical_weight_sum = float(sys.argv[17])
    dataset = json.loads(dataset_audit_path.read_text())
    loader = json.loads(loader_audit_path.read_text())
    failure = json.loads(failure_path.read_text())
    device = json.loads(device_path.read_text())
    assert dataset["accepted"] and dataset["source_public_only"]
    assert dataset["episodes"] == 15 and dataset["frames"] > 0
    assert loader["accepted"] and loader["right_gripper_closed_fraction"] > 0.99
    assert failure["completed_updates"] >= 0
    assert device["gpu_sha256"] == "1daaf04aa280349eda901dfe65869c7f7d30962dcf4322694ec4e8d2301ab2d9"
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "rules": {
            "real_submission": False,
            "hidden_or_reserved_evaluation_access": False,
            "training_data": "official/public successful demonstrations only",
        },
        "retry_of": failure,
        "resource_only_changes": {
            "parent_world_model_device": "cuda",
            "parent_world_model_release_cuda_cache_after_prediction": True,
            "parent_world_model_gpu_output_sha256_matches_original": True,
            "rollout_gc_and_empty_cache_before_model_reload": True,
            "pytorch_expandable_segments": expandable_segments,
            "policy_or_training_hyperparameter_change": False,
        },
        "initial_policy": {"name": initial_policy_name, "path": str(base_ckpt), "sha256": digest(base_ckpt)},
        "training": {
            "max_updates": 64,
            "episode_steps": 8,
            "group_size": 4,
            "total_envs": 8,
            "actor_seed": actor_seed,
            "actor_lr": actor_lr,
            "kl_beta_to_initial_policy": 0.1,
            "sft_loss_weight": 0.3,
            "sft_batch_size": 2,
            "single_active_arm": 1,
            "active_joint_weight": active_joint_weight,
            "active_gripper_weight": active_gripper_weight,
            "inactive_keep_weight": inactive_keep_weight,
            "physical_weight_sum": physical_weight_sum,
        },
        "next_gate": {
            "public_batch": "batch_00 (4 left, 12 right)",
            "thresholds": {"right_success": 2, "left_success": 2, "total_success": 4, "right_grasp_once": 10},
            "on_reject": "do not evaluate public batches 01-06",
        },
        "source_hashes": {
            "rollout_worker": digest(rollout_worker),
            "service_script": digest(service_script),
            "dataset_audit": digest(dataset_audit_path),
            "loader_audit": digest(loader_audit_path),
            "device_comparison": digest(device_path),
        },
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("V227_TERMINAL_RIGHT_SFT_RETRY_PREREGISTERED")


if __name__ == "__main__":
    main()
