#!/usr/bin/env python3
"""Preregister one-epoch mirror-balanced chunk8 SFT from frozen step5."""

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
    output, checkpoint, dataset_audit, conversion, routing_audit,
    actor_source, runner,
) = map(pathlib.Path, sys.argv[1:8])
candidate = sys.argv[8]
if output.exists():
    raise FileExistsError(output)
dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
routing = json.loads(routing_audit.read_text())
mirror_audit_path = pathlib.Path(converted["audit"])
mirror_audit = json.loads(mirror_audit_path.read_text())
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["episodes"] == converted["total_episodes"] == 65
assert dataset["frames"] == converted["total_frames"] == 9367
assert dataset["real_left_episodes"] == 25
assert dataset["effective_right_episodes"] == 40
assert mirror_audit["accepted"] and mirror_audit["forbidden_episodes_read"] == []
assert routing["public_data_only"] and routing["wrong_left_arm_routing_count"] == 7

ordinary_batches = 8
extra_batches = 2334
batch_size = 4
record = {
    "format": "strict-track2-v265-mirror-balanced-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "Repeated right-only SFT did not generalize beyond fifteen public right-arm scenes. "
        "A train40-only geometric augmentation balances 25 real left against 40 effective "
        "right trajectories. One shuffled epoch with an unstructured all-14-dimension loss "
        "should teach visual arm symmetry while rehearsing the frozen baseline's left behavior."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "fixed official public train40 plus deterministic train40 mirrors",
        "selection_data": "already-used public batch00 only",
    },
    "initial_policy": {
        "role": "frozen 51/128 step5 best baseline; no hidden reselection",
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
    "routing_evidence": {
        "path": str(routing_audit),
        "sha256": sha256(routing_audit),
        "wrong_left_arm_routing_count": routing["wrong_left_arm_routing_count"],
        "right_successes": routing["right_successes"],
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1465,
        "actor_lr": 2e-6,
        "kl_beta_to_initial_policy": 0.1,
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
        "structured_action_loss": False,
        "physical_action_dimensions": 14,
    },
    "public_gate": {
        "stage_r0_batch": "batch_00 (4 left, 12 right)",
        "thresholds": {"right_success": 4, "left_success": 3, "total_success": 7, "right_grasp": 10},
        "stage_r1_batches": ["batch_00", "batch_01"],
        "stage_r1_thresholds": {"right_success": 10, "left_success": 8, "total_success": 18, "grasp_once": 27},
        "on_reject": "do not access reserved/final 128",
    },
    "implementation": {
        "actor_source": str(actor_source), "actor_source_sha256": sha256(actor_source),
        "runner": str(runner), "runner_sha256": sha256(runner),
    },
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V265_MIRROR_BALANCED_SFT_PREREGISTERED")
