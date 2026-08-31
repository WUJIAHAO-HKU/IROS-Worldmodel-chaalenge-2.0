#!/usr/bin/env python3
"""Freeze the v348 learned-profile focused recursive gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v348_learned_profile_focused_train_seed1517_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    fit = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v347_public_temporal_residual_fit_seed1516_20260822"
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v346_learned_temporal_residual_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v348_learned_profile_focused_train.py",
        "profile": fit / "temporal_residual_profile.npz",
        "profile_manifest": fit / "temporal_residual_profile.manifest.json",
        "fit_report": fit / "fit_report.json",
        "v340_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822/train_recursive_gate_report.json",
    }
    payload = {
        "format": "strict-track2-v348-learned-profile-focused-train-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "frozen v347 learned five-frame profile; no coefficient changes",
        "scope": "same public-train alignments 0 and 4; exactly 484 recursive windows",
        "checks": "same frozen v342 focused thresholds; all required",
        "authorizes": "one public WM holdout gate only",
        "guards": {"public_train_only": True, "no_official_batch16_outcomes": True,
                   "no_hidden_or_final_data": True, "no_real_submission": True},
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
