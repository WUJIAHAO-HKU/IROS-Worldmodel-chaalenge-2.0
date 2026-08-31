#!/usr/bin/env python3
"""Audit and correct only the v209 manifest compatibility tag."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
OFF = BASE / "artifacts/strict_track2_official_20260810"
RELEASE = JOINT / "v209_v202_v208_public_arm_routed_release"
MANIFEST = RELEASE / "arm_routed_autoregressive_manifest.json"
REG = OFF / "run_registry/v209_v202_v208_public_arm_routed_release"
REGISTRATION = REG / "registration.json"
CORRECTION = REG / "manifest_compatibility_correction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if CORRECTION.exists():
        raise SystemExit("refusing to overwrite v209 compatibility correction")
    registration = json.loads(REGISTRATION.read_text())
    original_hash = sha256(MANIFEST)
    if original_hash != registration["manifest_sha256"]:
        raise ValueError("original v209 manifest no longer matches registration")
    document = json.loads(MANIFEST.read_text())
    if document.get("format") != "track2-arm-routed-autoregressive-release-v2":
        raise ValueError("unexpected original v209 manifest format")
    for name, expected in document["model_sha256"].items():
        if sha256(RELEASE / name / "model.pt") != expected:
            raise ValueError(f"expert hash mismatch: {name}")
    before = dict(document)
    document["format"] = "track2-arm-routed-autoregressive-release-v1"
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, MANIFEST)
    payload = {
        "format": "strict-track2-v209-manifest-compatibility-correction-v1",
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST),
        "original_manifest_sha256": original_hash,
        "corrected_manifest_sha256": sha256(MANIFEST),
        "only_changed_field": "format",
        "original_value": before["format"],
        "corrected_value": document["format"],
        "reason": "deployed backend explicitly accepts the established v1 schema tag",
        "model_sha256": document["model_sha256"],
        "model_bytes_changed": False,
        "hidden_or_final_evaluation_data": False,
        "real_submission_performed": False,
    }
    CORRECTION.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
