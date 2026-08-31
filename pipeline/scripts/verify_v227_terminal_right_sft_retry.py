#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path


run, checkpoint = map(Path, sys.argv[1:3])
expected_updates = int(sys.argv[3])
expected_physical_weight_sum = float(sys.argv[4]) if len(sys.argv) > 4 else 10.75
log = (run / "launcher.log").read_text(errors="replace")
fatals = [p for p in ("CUDA out of memory", "OutOfMemoryError", "Traceback (most recent call last)") if p in log]


def values(name: str) -> list[float]:
    return [float(v) for v in re.findall(rf"(?<![A-Za-z0-9_/]){re.escape(name)}=([-+0-9.eE]+)", log)]


names = ("sft_arm", "sft_physical_action_weight_sum", "sft_loss", "weighted_sft_loss", "actor/grad_norm", "actor/kl_loss")
metrics = {name: values(name) for name in names}
assert checkpoint.is_file() and checkpoint.stat().st_size > 8_000_000_000
assert zipfile.is_zipfile(checkpoint)
assert not fatals, fatals
for name in names[:-1]:
    assert len(metrics[name]) >= expected_updates, (name, len(metrics[name]))
    assert all(math.isfinite(v) for v in metrics[name][-expected_updates:])
assert all(v == 1.0 for v in metrics["sft_arm"][-expected_updates:])
assert all(abs(v - expected_physical_weight_sum) < 1e-9 for v in metrics["sft_physical_action_weight_sum"][-expected_updates:])
assert all(v > 0 for v in metrics["sft_loss"][-expected_updates:])
assert all(v > 0 for v in metrics["weighted_sft_loss"][-expected_updates:])
assert all(v > 0 for v in metrics["actor/grad_norm"][-expected_updates:])
sha = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
        sha.update(block)
report = {
    "accepted": True,
    "expected_updates": expected_updates,
    "metric_counts": {k: len(v) for k, v in metrics.items()},
    "metric_first": {k: (v[-expected_updates] if len(v) >= expected_updates else None) for k, v in metrics.items()},
    "metric_last": {k: (v[-1] if v else None) for k, v in metrics.items()},
    "checkpoint": str(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": sha.hexdigest(),
    "fatal_signatures": fatals,
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
audit = run / "audit"
audit.mkdir(exist_ok=True)
(audit / "terminal_right_sft_retry_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
(audit / f"{run.name.split('_')[0].upper()}_TERMINAL_RIGHT_SFT_RETRY_ACCEPTED").touch()
print(json.dumps(report, indent=2))
