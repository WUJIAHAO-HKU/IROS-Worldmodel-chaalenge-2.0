#!/usr/bin/env python3
"""Preregister v285: motion-weighted public right-terminal SFT from v278."""

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
    motion_test,
) = map(pathlib.Path, sys.argv[1:9])
candidate = sys.argv[9]
if output.exists():
    raise FileExistsError(f"refusing to overwrite {output}")

dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
temporal = json.loads(temporal_audit.read_text())
assert dataset["accepted"] and dataset["source_public_only"]
assert dataset["episodes"] == converted["episodes"] == 15
assert dataset["frames"] == converted["frames"] == 1292
assert temporal["public_data_only"] and temporal["deployed_horizon"] == 8
assert not temporal["reserved_final128_access"]
assert not converted["reserved_final128_access"]
assert not converted["real_competition_submission"]

ordinary_batches = 8
extra_batches = 315
batch_size = 4
record = {
    "format": "strict-track2-v285-motion-weighted-right-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "v278 already routes and grasps every public right-arm case, but the prior "
        "transition dataset stopped at close+48 exactly when successful demos begin "
        "their largest transport motion. One full public terminal epoch with deployed-"
        "horizon motion weighting should teach post-grasp lift/transport while retaining "
        "a small closed-gripper hold signal and preserving the inactive left arm."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "official public successful demonstrations only",
        "selection_data": "predeclared public development batch_00 only",
        "on_reject": "do not access batch_01 or reserved final128",
    },
    "initial_policy": {
        "role": "v278 completed conservative RL step10; frozen before this curriculum",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
    },
    "dataset": {
        "path": str(conversion.parent),
        "audit": str(dataset_audit),
        "audit_sha256": sha256(dataset_audit),
        "conversion": str(conversion),
        "conversion_sha256": sha256(conversion),
        "episodes": 15,
        "frames": 1292,
        "arm": "right",
        "window": "8 frames before right close through successful endpoint",
        "motion_rich_fraction_first8": temporal["global"]["motion_rich_fraction_first8"],
        "mean_padded_target_fraction_50": temporal["global"]["mean_padded_target_fraction_50"],
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1478,
        "actor_lr": 1e-6,
        "kl_beta_to_initial_policy": 0.1,
        "sft_batch_size": batch_size,
        "ordinary_sft_batches": ordinary_batches,
        "extra_sft_batches": extra_batches,
        "total_sft_batches": ordinary_batches + extra_batches,
        "sampled_sft_frames": (ordinary_batches + extra_batches) * batch_size,
        "nominal_dataset_epochs": (ordinary_batches + extra_batches) * batch_size / 1292,
        "use_action_chunk_loss": True,
        "supervised_temporal_horizon": 8,
        "motion_sample_weighting": {
            "active_arm": "right",
            "reference": 0.05,
            "minimum_weight": 0.05,
            "power": 1.0,
            "calibration_quantiles": {
                "q25": 0.00302,
                "q50": 0.02830,
                "q75": 0.10320,
                "q90": 0.18710,
            },
        },
        "structured_action_weights": {
            "right_joint": 4.0,
            "right_gripper": 1.0,
            "inactive_left": 2.0,
            "padding": 0.0,
        },
    },
    "public_gate": {
        "batch": "batch_00 (4 left, 12 right)",
        "thresholds": {
            "right_success": 4,
            "left_success": 3,
            "total_success": 7,
            "right_grasp": 10,
        },
    },
    "implementation": {
        "actor_source": str(actor_source),
        "actor_source_sha256": sha256(actor_source),
        "runner": str(runner),
        "runner_sha256": sha256(runner),
        "motion_contract_test": str(motion_test),
        "motion_contract_test_sha256": sha256(motion_test),
    },
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V285_MOTION_WEIGHTED_RIGHT_SFT_PREREGISTERED")
