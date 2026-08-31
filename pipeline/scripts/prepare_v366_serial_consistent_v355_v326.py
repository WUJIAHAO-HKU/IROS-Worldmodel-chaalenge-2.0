#!/usr/bin/env python3
"""Preregister the serial-consistent correction to rejected v365."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = J / "v366_serial_consistent_v355_v326_seed1530_20260822"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists():
        raise FileExistsError("refusing overwrite of v366")
    paths = {
        "v365_registration": J / "v365_v355_v326_terminal_hybrid_seed1529_20260822/release_registration.json",
        "v365_failed_contract": J / "v365_v355_v326_terminal_hybrid_seed1529_20260822/contract_report.json",
        "runtime": ROOT / "pipeline/wam_pipeline/v366_serial_consistent_v355_v326_runtime.py",
        "backend_factory": ROOT / "pipeline/wam_pipeline/backends.py",
        "contract": ROOT / "pipeline/scripts/test_v366_serial_consistent_v355_v326.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    REG = json.loads(paths["v365_failed_contract"].read_text())
    if REG.get("passed") is not False:
        raise RuntimeError("v366 requires the frozen v365 contract failure")
    RUN.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v366-serial-consistent-v355-v326-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "v365 semantics with predict_batch defined as exact per-sample predict stacking",
        "fixed_contract": {"serial_batch_pixel_bit_exact": True, "left_parent_bit_exact": True, "causal_counterfactuals": True, "shape_dtype": True, "internal_speed_proxy": "removed before measurement", "replacement_throughput_gate": "complete HTTP batch8 acceptance under official recommended timeout"},
        "reason": "v365 failed numerical batch equivalence; service latency is the deployment-relevant throughput measure",
        "sequence": ["numeric contract", "public causal/reward audit", "HTTP acceptance", "training-mode rollout-only128"],
        "evidence_sha256": {key: sha(path) for key, path in paths.items()},
        "guards": {"public_data_only": True, "policy_modified": False, "hidden_or_final_data": False, "real_submission": False},
    }
    (RUN / "release_registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(RUN)


if __name__ == "__main__":
    main()
