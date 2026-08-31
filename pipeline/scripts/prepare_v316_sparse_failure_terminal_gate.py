#!/usr/bin/env python3
"""Create a hash-corrected preregistration after v315 import-only drift."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
OLD = J / "v315_v311_sparse_failure_terminal_gate_seed1488_20260822"
N = "v316_v315_hash_corrected_sparse_failure_gate_seed1489_20260822"
RUN = J / N
REG = O / "run_registry" / N


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    source = OLD / "release_registration.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text())
    paths = {
        key: Path(value["path"])
        for key, value in payload["evidence"].items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload["format"] = "strict-track2-v316-hash-corrected-sparse-failure-preregistration-v1"
    payload["registered_at"] = datetime.now(timezone.utc).isoformat()
    payload["candidate"]["model_version"] = "track2-v315-sparse-failure-terminal-v271-r1"
    payload["registration_repair"] = {
        "invalidated_predecessor": str(OLD),
        "reason": "contract test import path corrected before any model execution, reward audit, or RL",
        "behavior_change": False,
        "threshold_change": False,
        "audit_change": False,
    }
    payload["evidence"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
