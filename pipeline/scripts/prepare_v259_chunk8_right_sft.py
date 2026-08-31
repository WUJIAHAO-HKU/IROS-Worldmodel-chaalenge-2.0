#!/usr/bin/env python3
"""Preregister the public-only, deployment-horizon-aligned v259 candidate."""

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
    temporal_audit,
    actor_source,
    runner,
) = map(pathlib.Path, sys.argv[1:8])
candidate = sys.argv[8]
if output.exists():
    raise FileExistsError(f"refusing to overwrite {output}")

dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
temporal = json.loads(temporal_audit.read_text())
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["episodes"] == converted["episodes"] == 15
assert dataset["frames"] == converted["frames"] == 1292
assert temporal["public_data_only"]
assert temporal["deployed_horizon"] == 8
assert temporal["current_uncropped_sft_horizon"] == 50
assert not converted["reserved_final128_access"]
assert not converted["real_competition_submission"]

ordinary_batches = 8
extra_batches = 128
batch_size = 2
record = {
    "format": "strict-track2-v259-chunk8-right-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "The prior SFT objective averaged all 50 predicted temporal positions "
        "while deployment executes only action_chunk=8. Cropping supervision to "
        "the deployed horizon removes distant and endpoint-padded targets."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "official public successful demonstrations only",
        "selection_data": "predeclared public development seeds only",
    },
    "initial_policy": {
        "role": "previously frozen 51/128 best baseline; no hidden reselection",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
    },
    "dataset": {
        "path": str(conversion.parent),
        "audit": str(dataset_audit),
        "conversion": str(conversion),
        "episodes": 15,
        "frames": 1292,
        "arm": "right",
        "window": "8 frames before right close through successful endpoint",
        "temporal_audit": str(temporal_audit),
        "temporal_audit_sha256": sha256(temporal_audit),
        "motion_rich_fraction_first8": temporal["global"][
            "motion_rich_fraction_first8"
        ],
        "mean_padded_target_fraction_50": temporal["global"][
            "mean_padded_target_fraction_50"
        ],
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1461,
        "actor_lr": 5e-6,
        "kl_beta_to_initial_policy": 0.1,
        "sft_batch_size": batch_size,
        "ordinary_sft_batches": ordinary_batches,
        "extra_sft_batches": extra_batches,
        "total_sft_batches": ordinary_batches + extra_batches,
        "sampled_sft_frames": (ordinary_batches + extra_batches) * batch_size,
        "nominal_dataset_epochs": (
            (ordinary_batches + extra_batches) * batch_size / converted["frames"]
        ),
        "use_action_chunk_loss": True,
        "supervised_temporal_horizon": 8,
        "model_prediction_horizon": 50,
        "extra_sft_loss_weight": 1.0,
        "structured_action_weights": {
            "right_joint": 1.0,
            "right_gripper": 3.0,
            "inactive_left": 0.25,
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
        "on_reject": "do not access reserved/final 128; revise on public data only",
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
print("V259_CHUNK8_RIGHT_SFT_PREREGISTERED")
