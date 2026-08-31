#!/usr/bin/env python3
"""Freeze the focused v344 alpha=0.75 train gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v344_focused_train_recursive_seed1513_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v344_conservative_temporal_reanchor_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v344_focused_train_recursive.py",
        "v342_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v342_focused_train_recursive_seed1511_20260822/focused_train_gate_report.json",
        "v343_holdout_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v343_v342_public_holdout_recursive_seed1512_20260822/holdout_recursive_gate_report.json",
    }
    payload = {
        "format": "strict-track2-v344-focused-train-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "parent": "v334", "recursive_gate": "v339",
            "reanchor_alpha": 0.75, "phase_max_offset": 16,
            "feature_workers_max": 8,
        },
        "scope": "same frozen public-train alignments 0 and 4; exactly 484 windows",
        "checks": "same v342 focused thresholds; all required",
        "authorizes": "public WM holdout gate only",
        "guards": {
            "public_train_only": True, "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True, "no_real_submission": True,
        },
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
