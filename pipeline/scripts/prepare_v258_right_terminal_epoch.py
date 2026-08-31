#!/usr/bin/env python3
"""Preregister one public-only right-terminal SFT epoch from the frozen baseline."""

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


output, checkpoint, dataset_audit, conversion, actor_source, runner = map(
    pathlib.Path, sys.argv[1:7]
)
candidate = (
    sys.argv[7]
    if len(sys.argv) > 7
    else "v258_v169step5_rightterminal_sft1epoch_h8_step1_extra638_lr5e6_seed1460_20260819"
)
retry_of = pathlib.Path(sys.argv[8]) if len(sys.argv) > 8 else None
if output.exists():
    raise FileExistsError(f"refusing to overwrite {output}")
if retry_of is not None and not retry_of.is_file():
    raise FileNotFoundError(f"retry preregistration is missing: {retry_of}")

dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["episodes"] == converted["episodes"] == 15
assert dataset["frames"] == converted["frames"] == 1292
assert not converted["reserved_final128_access"]
assert not converted["real_competition_submission"]

record = {
    "format": "strict-track2-v258-right-terminal-extra-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "official public successful demonstrations only",
        "selection_data": "predeclared public development seeds only",
    },
    "initial_policy": {
        "role": "previously frozen best baseline; not reselected in this run",
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
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1460,
        "actor_lr": 5e-6,
        "kl_beta_to_initial_policy": 0.1,
        "sft_batch_size": 2,
        "ordinary_sft_batches": 8,
        "extra_sft_batches": 638,
        "total_sft_batches": 646,
        "total_sft_frames": 1292,
        "dataset_epochs": 1.0,
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
        "on_reject": "do not access the reserved/final 128; adjust public-only SFT",
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
if retry_of is not None:
    record["retry_of"] = {
        "preregistration": str(retry_of),
        "preregistration_sha256": sha256(retry_of),
        "reason": (
            "implementation-only retry after the prior extra-SFT backward failed "
            "before checkpointing or evaluation; training data, seed, hyperparameters, "
            "and predeclared public gates are unchanged"
        ),
    }
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V258_RIGHT_TERMINAL_EPOCH_PREREGISTERED")
