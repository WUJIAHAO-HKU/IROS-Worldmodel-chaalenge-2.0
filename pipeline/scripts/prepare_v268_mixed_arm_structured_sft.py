#!/usr/bin/env python3
"""Preregister per-example mixed-arm structured SFT from v265."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


(
    output, checkpoint, dataset_audit, conversion, inference_audit_path,
    actor_source, runner,
) = map(pathlib.Path, sys.argv[1:8])
candidate = sys.argv[8]
if output.exists():
    raise FileExistsError(output)
dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
inference = json.loads(inference_audit_path.read_text())
off = output.parents[2]
failure_path = off / "diagnostics" / "v267_v266_right_terminal_failure_20260819.json"
failure = json.loads(failure_path.read_text())
mirror_audit_path = pathlib.Path(converted["audit"])
mirror_audit = json.loads(mirror_audit_path.read_text())
openpi_source = (
    actor_source.parents[2]
    / "models" / "embodiment" / "openpi" / "openpi_action_model.py"
)
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["episodes"] == converted["total_episodes"] == 65
assert dataset["frames"] == converted["total_frames"] == 9367
assert dataset["real_left_episodes"] == 25
assert dataset["effective_right_episodes"] == 40
assert mirror_audit["accepted"] and mirror_audit["forbidden_episodes_read"] == []
assert inference["accepted"] and inference["total_frames"] == 9367
assert inference["left_accuracy"] == inference["right_accuracy"] == 1.0
assert failure["wrong_left_arm_routing_count"] == 7
assert failure["right_never_closed_count"] == 5
assert "build_mixed_track2_sft_action_loss_weights" in actor_source.read_text()
assert "reduce_sft_action_loss" in openpi_source.read_text()

ordinary_batches = 8
extra_batches = 2334
batch_size = 4
record = {
    "format": "strict-track2-v268-mixed-arm-structured-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "The equivalent v266 action capture shows seven of twelve public right scenes still "
        "route more motion through the left arm and five never close the right gripper. "
        "Equal 14D loss does not sufficiently suppress inactive-arm motion. Infer the active "
        "demonstration arm per sample using official quantile zero points (audited 9367/9367), "
        "then weight active joints 1, active gripper 3, and every inactive-arm dimension 2."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "fixed official public train40 plus deterministic train40 mirrors",
        "selection_data": "already-used public batch00 action capture only",
    },
    "initial_policy": {
        "role": "v265 public-R0 diagnostic checkpoint; derived from frozen step5 and stronger than v266",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
    },
    "dataset": {
        "path": str(conversion.parent),
        "audit": str(dataset_audit),
        "audit_sha256": sha256(dataset_audit),
        "conversion": str(conversion),
        "conversion_sha256": sha256(conversion),
        "mirror_audit": str(mirror_audit_path),
        "mirror_audit_sha256": sha256(mirror_audit_path),
        "episodes": 65,
        "frames": 9367,
        "real_left_episodes": 25,
        "real_right_episodes": 15,
        "mirrored_left_to_right_episodes": 25,
        "effective_right_episodes": 40,
    },
    "diagnostic_evidence": {
        "action_capture_failure": str(failure_path),
        "action_capture_failure_sha256": sha256(failure_path),
        "wrong_left_arm_routing_count": failure["wrong_left_arm_routing_count"],
        "right_never_closed_count": failure["right_never_closed_count"],
        "inference_audit": str(inference_audit_path),
        "inference_audit_sha256": sha256(inference_audit_path),
        "inferred_arm_accuracy": inference["overall_accuracy"],
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1468,
        "actor_lr": 1e-6,
        "kl_beta_to_initial_policy": 0.2,
        "sft_batch_size": batch_size,
        "ordinary_sft_batches": ordinary_batches,
        "extra_sft_batches": extra_batches,
        "total_sft_batches": ordinary_batches + extra_batches,
        "sampled_sft_frames": (ordinary_batches + extra_batches) * batch_size,
        "nominal_dataset_epochs": (
            (ordinary_batches + extra_batches) * batch_size / converted["total_frames"]
        ),
        "use_action_chunk_loss": True,
        "supervised_temporal_horizon": 8,
        "model_prediction_horizon": 50,
        "ordinary_sft_loss_weight": 0.3,
        "extra_sft_loss_weight": 1.0,
        "mixed_arm_structured_action_loss": True,
        "active_joint_weight": 1.0,
        "active_gripper_weight": 3.0,
        "inactive_keep_weight": 2.0,
        "gripper_activity_weight": 0.25,
        "expected_action_weight_sum": 23.0,
        "model_action_dimensions": 32,
        "physical_action_dimensions": 14,
        "padded_model_action_dimensions_ignored": True,
    },
    "public_gate": {
        "stage_r0_batch": "batch_00 (4 left, 12 right)",
        "thresholds": {"right_success": 4, "left_success": 3, "total_success": 7, "right_grasp": 10},
        "stage_r1_batches": ["batch_00", "batch_01"],
        "stage_r1_thresholds": {"right_success": 10, "left_success": 8, "total_success": 18, "grasp_once": 27},
        "on_reject": "do not access reserved/final 128",
    },
    "implementation": {
        "actor_source": str(actor_source),
        "actor_source_sha256": sha256(actor_source),
        "openpi_source": str(openpi_source),
        "openpi_source_sha256": sha256(openpi_source),
        "runner": str(runner),
        "runner_sha256": sha256(runner),
    },
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V268_MIXED_ARM_STRUCTURED_SFT_PREREGISTERED")
