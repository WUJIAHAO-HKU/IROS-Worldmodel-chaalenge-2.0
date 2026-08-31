#!/usr/bin/env python3
"""Freeze the focused v342 train gate before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v342_focused_train_recursive_seed1511_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v342_temporal_blended_public_reanchor_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v342_focused_train_recursive.py",
        "v340_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822/train_recursive_gate_report.json",
        "v341_holdout_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v341_v340_public_holdout_recursive_seed1510_20260822/holdout_recursive_gate_report.json",
    }
    payload = {
        "format": "strict-track2-v342-focused-train-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "parent": "v334", "recursive_gate": "v339",
            "reanchor_alpha": 0.90, "phase_max_offset": 16,
            "feature_workers_max": 8,
        },
        "scope": "15 public-train right episodes, frozen alignments 0 and 4, exactly 484 windows",
        "teacher_equivalence": "zero gate activations plus cross-instance RGB/temporal metric delta <=0.01",
        "checks_all_required": {
            "direct_count": ">=20", "direct_rgb_temporal": "ratios <=0.90",
            "direct_reward": "error ratio <=1.10, mean>=.90, hit>=.85",
            "all_nonregression": "RGB/temporal/reward ratios <=1.02",
        },
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
