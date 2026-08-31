#!/usr/bin/env python3
"""Freeze v340 training-only recursive gate before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "v338_runtime": ROOT / "pipeline/wam_pipeline/v338_recursive_public_reanchor_runtime.py",
        "v340_runtime": ROOT / "pipeline/wam_pipeline/v340_high_specificity_public_reanchor_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v340_train_recursive_reanchor.py",
        "v339_gate": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v339_high_specificity_recursive_ood_gate_seed1508_20260822/recursive_ood_gate.npz",
    }
    payload = {
        "format": "strict-track2-v340-train-recursive-reanchor-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "parent": "v334",
            "recursive_gate": "v339 threshold 0.925",
            "reanchor_alpha": 1.0,
            "target": "action/visual matched eligible public phase row",
            "target_phase_max_offset": 16,
            "changed_frames": [3, 4, 5, 6, 7],
        },
        "coverage": "all 1926 windows exactly once via 8 alignment chains in 15 public-train right episodes",
        "checks_all_required": {
            "teacher": "zero reanchors and bit-exact v334",
            "changed_count": ">=100",
            "changed_rgb_and_temporal_ratio": "<=0.90",
            "changed_reward_error_ratio": "<=1.10",
            "changed_reward_mean_hit": ">=.90 and hit>=.85",
            "all_window_nonregression_ratios": "<=1.02",
        },
        "authorizes_rl": False,
        "guards": {
            "public_train_only": True,
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
