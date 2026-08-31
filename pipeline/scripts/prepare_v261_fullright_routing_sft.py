#!/usr/bin/env python3
"""Preregister full-trajectory right-arm routing correction from public data."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


(
    output,
    checkpoint,
    dataset_audit,
    conversion,
    routing_audit,
    actor_source,
    runner,
) = map(pathlib.Path, sys.argv[1:8])
candidate = sys.argv[8]
if output.exists():
    raise FileExistsError(output)

dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
routing = json.loads(routing_audit.read_text())
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["frames"] == converted["num_frames"] == 2091
assert dataset["episodes"] == converted["num_episodes"] == 15
assert converted["arm_filter"] == "arm1"
assert routing["public_data_only"]
assert routing["right_successes"] == 0
assert routing["wrong_left_arm_routing_count"] >= 6
assert not routing["reserved_final128_access"]

ordinary_batches = 8
extra_batches = 512
batch_size = 2
record = {
    "format": "strict-track2-v261-fullright-routing-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "Terminal-only SFT does not expose initial arm-selection contexts. Complete right-arm "
        "demonstrations plus equal inactive-left supervision should correct wrong-arm routing; "
        "an 8x right-gripper weight should make contact become an actual closed grasp."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "frozen official public train-split demonstrations only",
        "selection_data": "already-used public batch00 only",
    },
    "initial_policy": {
        "role": "previously frozen 51/128 best baseline; no hidden reselection",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
    },
    "dataset": {
        "path": str(conversion.parent),
        "audit": str(dataset_audit),
        "audit_sha256": sha256(dataset_audit),
        "conversion": str(conversion),
        "episodes": 15,
        "frames": 2091,
        "arm": "right",
        "window": "complete successful trajectory",
    },
    "routing_evidence": {
        "path": str(routing_audit),
        "sha256": sha256(routing_audit),
        "right_episodes": routing["right_episodes"],
        "wrong_left_arm_routing_count": routing["wrong_left_arm_routing_count"],
        "right_never_closed_count": routing["right_never_closed_count"],
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1462,
        "actor_lr": 5e-6,
        "kl_beta_to_initial_policy": 0.1,
        "sft_batch_size": batch_size,
        "ordinary_sft_batches": ordinary_batches,
        "extra_sft_batches": extra_batches,
        "total_sft_batches": ordinary_batches + extra_batches,
        "sampled_sft_frames": (ordinary_batches + extra_batches) * batch_size,
        "nominal_dataset_epochs": (
            (ordinary_batches + extra_batches) * batch_size / converted["num_frames"]
        ),
        "use_action_chunk_loss": True,
        "supervised_temporal_horizon": 8,
        "model_prediction_horizon": 50,
        "extra_sft_loss_weight": 1.0,
        "structured_action_weights": {
            "right_joint": 1.0,
            "right_gripper": 8.0,
            "inactive_left": 1.0,
            "padding": 0.0,
        },
    },
    "public_gate": {
        "stage_r0_batch": "batch_00 (4 left, 12 right)",
        "thresholds": {
            "right_success": 4,
            "left_success": 3,
            "total_success": 7,
            "right_grasp": 10,
        },
        "stage_r1_batches": ["batch_00", "batch_01"],
        "stage_r1_thresholds": {
            "right_success": 10,
            "left_success": 8,
            "total_success": 18,
            "grasp_once": 27,
        },
        "on_reject": "do not access reserved/final 128",
    },
    "implementation": {
        "actor_source": str(actor_source),
        "actor_source_sha256": sha256(actor_source),
        "runner": str(runner),
        "runner_sha256": sha256(runner),
    },
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V261_FULLRIGHT_ROUTING_SFT_PREREGISTERED")
