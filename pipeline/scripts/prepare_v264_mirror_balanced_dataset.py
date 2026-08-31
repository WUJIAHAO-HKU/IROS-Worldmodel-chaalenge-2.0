#!/usr/bin/env python3
"""Preregister the public train40 mirror-balanced SFT dataset build."""

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


output, mirror_audit, builder, auditor = map(pathlib.Path, sys.argv[1:5])
dataset = pathlib.Path(sys.argv[5])
if output.exists():
    raise FileExistsError(output)
audit = json.loads(mirror_audit.read_text())
assert audit["accepted"] and audit["forbidden_episodes_read"] == []
assert audit["left_episode_count"] == 25 and audit["right_episode_count"] == 15
record = {
    "format": "strict-track2-v264-public-mirror-balanced-dataset-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hypothesis": (
        "The fixed public train40 contains only 15 right-arm versus 25 left-arm successful "
        "demonstrations. Horizontally flipping each public left image and applying the exact "
        "involutive Aloha joint mirror supplies 25 additional right-arm trajectories without "
        "accessing validation, local-test, hidden, or real-evaluation data."
    ),
    "source_policy": {
        "real_train_episodes": 40,
        "mirrored_left_train_episodes": 25,
        "expected_total_episodes": 65,
        "expected_real_left": 25,
        "expected_effective_right": 40,
    },
    "mirror_audit": {
        "path": str(mirror_audit),
        "sha256": sha256(mirror_audit),
        "action_rmse_plain": audit["trajectory_alignment_error"]["plain_swap"]["rmse"],
        "action_rmse_mirrored": audit["trajectory_alignment_error"]["signed_mirror"]["rmse"],
        "image_mae_plain": audit["mean_image_alignment"]["plain_mae"],
        "image_mae_flipped": audit["mean_image_alignment"]["horizontal_flip_mae"],
    },
    "dataset_output": str(dataset),
    "implementation": {
        "builder": str(builder), "builder_sha256": sha256(builder),
        "auditor": str(auditor), "auditor_sha256": sha256(auditor),
    },
    "rules": {
        "training_split_only": True,
        "forbidden_episode_ids_read": [],
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    },
}
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V264_MIRROR_BALANCED_DATASET_PREREGISTERED")
