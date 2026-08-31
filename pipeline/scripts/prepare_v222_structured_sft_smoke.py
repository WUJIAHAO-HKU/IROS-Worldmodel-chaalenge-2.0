#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


out, base_ckpt, dataset, worker, model, launcher = map(Path, sys.argv[1:7])
files = sorted(path for path in dataset.rglob("*") if path.is_file())
dataset_digest = hashlib.sha256()
for path in files:
    dataset_digest.update(str(path.relative_to(dataset)).encode())
    dataset_digest.update(bytes.fromhex(sha256(path)))

report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "public-data-only structured right-arm SFT one-update smoke",
    "real_competition_submission": False,
    "reserved_final128_access": False,
    "base_checkpoint": str(base_ckpt),
    "base_checkpoint_sha256": sha256(base_ckpt),
    "sft_dataset": str(dataset),
    "sft_dataset_tree_sha256": dataset_digest.hexdigest(),
    "training": {
        "updates": 1,
        "active_arm": 1,
        "structured_action_loss": True,
        "active_joint_weight": 1.0,
        "active_gripper_weight": 3.0,
        "inactive_keep_weight": 0.25,
        "expected_physical_weight_sum": 10.75,
        "padded_dim_weight": 0.0,
        "sft_loss_weight": 0.1,
        "sft_batch_size": 2,
        "actor_lr": 1e-5,
        "kl_beta": 0.1,
    },
    "source_sha256": {
        "fsdp_actor_worker.py": sha256(worker),
        "openpi_action_model.py": sha256(model),
        "run_strict_track2_conservative_kl.sh": sha256(launcher),
    },
    "acceptance": [
        "one valid checkpoint",
        "sft_arm=1",
        "sft_physical_action_weight_sum=10.75",
        "finite SFT loss and nonzero finite grad norm",
        "no OOM or traceback",
    ],
}
out.parent.mkdir(parents=True, exist_ok=False)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
