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


out, checkpoint, training_report, batch, prior_gate = map(Path, sys.argv[1:6])
training = json.loads(training_report.read_text())
gate = json.loads(prior_gate.read_text())
assert training["accepted"] is True
assert gate["accepted"] is False
record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "protocol": f"{out.parent.name} evidence-only action/state capture on the already-used public batch00",
    "candidate_checkpoint": str(checkpoint),
    "checkpoint_sha256": training.get("checkpoint_sha256", digest(checkpoint)),
    "batch_sha256": digest(batch),
    "prior_gate_counts": gate["counts"],
    "capture": {
        "policy_action_chunks": True,
        "reset_and_post_step_observations": True,
        "policy_or_environment_modification": False,
        "expected_equivalence": "all aggregate metrics and bitmasks equal the prior v223 batch00 run",
    },
    "selection_use": "diagnose terminal motion only; no extra seeds",
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
out.parent.mkdir(parents=True, exist_ok=False)
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
