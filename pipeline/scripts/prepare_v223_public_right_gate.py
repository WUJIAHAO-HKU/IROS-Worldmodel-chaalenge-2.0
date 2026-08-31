#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


out, checkpoint, training_report, manifest, batch = map(Path, sys.argv[1:6])
training = json.loads(training_report.read_text())
assert training["accepted"] is True
assert training["checkpoint"] == str(checkpoint)
assert training["checkpoint_bytes"] == checkpoint.stat().st_size
record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "variant": "v223_v169_structured_rightsft_kl_step24_seed1423",
    "purpose": "public batch00 gate after accepted structured arm1 training",
    "selection_uses_policy_outcomes": False,
    "batch": 0,
    "episode_composition": {"left": 4, "right": 12, "total": 16},
    "thresholds": {
        "right_success": 1,
        "left_success": 2,
        "total_success": 3,
        "grasp_once": 12,
    },
    "checkpoint_sha256": training["checkpoint_sha256"],
    "manifest_sha256": digest(manifest),
    "batch_sha256": digest(batch),
    "real_submission": False,
    "reserved_final128_used": False,
    "on_reject": "do not evaluate public batches 01-06; revise public-data training",
}
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
