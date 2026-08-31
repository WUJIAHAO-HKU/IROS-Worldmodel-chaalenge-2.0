#!/usr/bin/env python3
"""Record the non-model v217 fix for a negative missing-seed sentinel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v217_v216_online_recursive_service_gate_seed1416"
RUN = JOINT / NAME
REG = OFF / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = REG / "protocol_seed_amendment.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    prereg = json.loads((REG / "preregistration.json").read_text())
    exporter = BASE / "pipeline/scripts/export_v217_service_recursive_cache.py"
    runtime = BASE / "pipeline/wam_pipeline/v216_public_knn_blend_runtime.py"
    parent = BASE / "pipeline/wam_pipeline/autoregressive_unet_runtime.py"
    resume = BASE / "pipeline/scripts/resume_v217_after_seed_fix.sh"
    old_hash = prereg["implementation"][str(exporter)]
    new_hash = sha256(exporter)
    if old_hash == new_hash:
        raise ValueError("exporter did not change")
    parent_source = parent.read_text()
    if "del seeds, instructions" not in parent_source:
        raise ValueError("cannot prove native batch predictions ignore seed")
    success_baseline = RUN / "audit/public_success_baseline.npz"
    with np.load(success_baseline, allow_pickle=False) as values:
        source_seeds = values["synthetic_seed"].astype(np.int64)
    if not np.all(source_seeds == -1):
        raise ValueError("unexpected public-success seed sentinel")
    if (RUN / "audit/public_success_candidate.npz").exists():
        raise ValueError("cannot amend after a candidate cache was produced")
    service_acceptance = json.loads((RUN / "audit/service_acceptance.json").read_text())
    if service_acceptance.get("passed") is not True:
        raise ValueError("service acceptance did not pass before the export failure")
    amendment = {
        "format": "strict-track2-v217-protocol-seed-amendment-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "official API rejected local missing-seed sentinel -1 before any candidate frame was produced",
        "old_exporter_sha256": old_hash,
        "new_exporter_sha256": new_hash,
        "resume_script_sha256": sha256(resume),
        "runtime_sha256_unchanged": sha256(runtime),
        "model_or_retrieval_change": False,
        "gate_or_data_split_change": False,
        "fix": "replace only negative missing source seeds with protocol-valid zero",
        "native_batch_seed_ignored_evidence": {
            "source": str(parent),
            "source_sha256": sha256(parent),
            "statement": "predict_batch deletes seeds and instructions before deterministic inference",
        },
        "candidate_cache_existed_before_amendment": False,
        "hidden_or_final_data": False,
        "real_submission": False,
    }
    output.write_text(json.dumps(amendment, indent=2) + "\n")
    (RUN / "protocol_seed_amendment.json").write_text(json.dumps(amendment, indent=2) + "\n")
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
