#!/usr/bin/env python3
"""Freeze the v342 final public-WM holdout gate before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v343_v342_public_holdout_recursive_seed1512_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v342_temporal_blended_public_reanchor_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v343_v342_public_holdout_recursive.py",
        "focused_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v342_focused_train_recursive_seed1511_20260822/focused_train_gate_report.json",
        "v335_baseline_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v335_v334_all_offset_recursive_gate_seed1505_20260822/audit/recursive_stability_report.json",
    }
    payload = {
        "format": "strict-track2-v343-v342-public-holdout-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "frozen v342 alpha=0.90, phase max offset=16, v339 OOD gate",
        "coverage": "all 512 public WM holdout windows exactly once over 8 alignment chains",
        "teacher_equivalence": "zero runtime gate activations and metric max delta <=0.01 across model reload",
        "checks_all_required": {
            "direct_count_each_split": ">=10",
            "direct_rgb_temporal_each_split": "ratios <=0.90",
            "direct_reward_each_split": "error ratio<=1.10, mean>=.90, hit>=.85",
            "all_nonregression_each_split": "RGB/temporal/reward ratios<=1.02",
        },
        "authorizes": "service acceptance only",
        "guards": {
            "public_world_model_holdout_only": True,
            "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True, "no_real_submission": True,
        },
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
