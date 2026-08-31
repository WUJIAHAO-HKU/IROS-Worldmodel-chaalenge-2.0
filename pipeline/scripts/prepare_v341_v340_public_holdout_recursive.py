#!/usr/bin/env python3
"""Freeze v340 public-holdout recursive gate before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v341_v340_public_holdout_recursive_seed1510_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "v340_runtime": ROOT / "pipeline/wam_pipeline/v340_high_specificity_public_reanchor_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v341_v340_public_holdout_recursive.py",
        "v340_train_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822/train_recursive_gate_report.json",
        "v335_baseline_report": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v335_v334_all_offset_recursive_gate_seed1505_20260822/audit/recursive_stability_report.json",
    }
    payload = {
        "format": "strict-track2-v341-v340-public-holdout-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "frozen v340",
        "coverage": "all 512 public holdout windows exactly once over 8 alignment chains",
        "baseline": "frozen v335 report's bit-identical v334 rows",
        "checks_all_required": {
            "teacher": "bit-exact v334",
            "direct_count_each_split": ">=10",
            "direct_rgb_and_temporal_ratio_each_split": "<=0.90",
            "direct_reward_error_ratio_each_split": "<=1.10",
            "direct_reward_mean_hit_each_split": ">=.90 and hit>=.85",
            "all_window_nonregression_each_split": "RGB/temporal/reward error ratios <=1.02",
        },
        "authorizes": "service acceptance only, never RL directly",
        "guards": {
            "public_world_model_holdout_only": True,
            "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True,
            "no_real_submission": True,
        },
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
