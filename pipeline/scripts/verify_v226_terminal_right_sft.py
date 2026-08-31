#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path


run = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
expected_updates = int(sys.argv[3])
log = (run / "launcher.log").read_text(errors="replace")

fatals = [
    pattern
    for pattern in ("CUDA out of memory", "OutOfMemoryError", "Traceback (most recent call last)")
    if pattern in log
]


def values(name: str) -> list[float]:
    return [
        float(item)
        for item in re.findall(rf"(?<![A-Za-z0-9_/]){re.escape(name)}=([-+0-9.eE]+)", log)
    ]


metric_names = (
    "sft_arm",
    "sft_physical_action_weight_sum",
    "sft_loss",
    "weighted_sft_loss",
    "actor/grad_norm",
    "actor/kl_loss",
)
metrics = {name: values(name) for name in metric_names}
assert checkpoint.is_file() and checkpoint.stat().st_size > 8_000_000_000
assert zipfile.is_zipfile(checkpoint)
assert not fatals, fatals
for name in metric_names[:-1]:
    assert len(metrics[name]) >= expected_updates, (name, len(metrics[name]))
    assert all(math.isfinite(value) for value in metrics[name][-expected_updates:])
assert all(value == 1.0 for value in metrics["sft_arm"][-expected_updates:])
assert all(abs(value - 10.75) < 1e-9 for value in metrics["sft_physical_action_weight_sum"][-expected_updates:])
assert all(value > 0 for value in metrics["sft_loss"][-expected_updates:])
assert all(value > 0 for value in metrics["weighted_sft_loss"][-expected_updates:])
assert all(value > 0 for value in metrics["actor/grad_norm"][-expected_updates:])

digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        digest.update(block)
report = {
    "accepted": True,
    "expected_updates": expected_updates,
    "metric_counts": {name: len(items) for name, items in metrics.items()},
    "metric_first": {name: items[-expected_updates] if len(items) >= expected_updates else None for name, items in metrics.items()},
    "metric_last": {name: items[-1] if items else None for name, items in metrics.items()},
    "checkpoint": str(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": digest.hexdigest(),
    "fatal_signatures": fatals,
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
audit = run / "audit"
audit.mkdir(exist_ok=True)
(audit / "terminal_right_sft_training_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
(audit / "V226_TERMINAL_RIGHT_SFT_TRAINING_ACCEPTED").touch()
print(json.dumps(report, indent=2))
