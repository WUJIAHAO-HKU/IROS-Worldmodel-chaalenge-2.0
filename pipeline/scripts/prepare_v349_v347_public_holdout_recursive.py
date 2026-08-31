#!/usr/bin/env python3
"""Freeze the final v347 learned-profile public-WM holdout gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v349_v347_public_holdout_recursive_seed1518_20260822"


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
        "audit": ROOT / "pipeline/scripts/audit_v349_v347_public_holdout_recursive.py",
        "profile": fit / "temporal_residual_profile.npz",
        "fit_report": fit / "fit_report.json",
        "v348_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v348_learned_profile_focused_train_seed1517_20260822/focused_train_gate_report.json",
        "v335_baseline_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v335_v334_all_offset_recursive_gate_seed1505_20260822/audit/recursive_stability_report.json",
    }
    payload = {
        "format": "strict-track2-v349-v347-public-holdout-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "frozen v347 learned profile; no post-fit coefficient adjustment",
        "coverage": "all 512 public WM holdout windows exactly once over 8 alignment chains",
        "checks": "same frozen v343/v345 thresholds; all required",
        "authorizes": "service acceptance only",
        "guards": {"public_world_model_holdout_only": True,
                   "no_official_batch16_outcomes": True,
                   "no_hidden_or_final_data": True, "no_real_submission": True},
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
