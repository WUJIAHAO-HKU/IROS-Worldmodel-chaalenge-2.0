#!/usr/bin/env python3
"""Freeze the v337 public-holdout generalization gate before execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v337_public_recursive_ood_gate_seed1507_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    output = RUN / "holdout_preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "audit": ROOT / "pipeline/scripts/audit_v337_recursive_ood_holdout.py",
        "feature_contract": ROOT / "pipeline/wam_pipeline/v337_public_recursive_ood_gate.py",
        "train_preregistration": RUN / "release_registration.json",
        "training_report": RUN / "training_report.json",
        "recursive_ood_gate": RUN / "recursive_ood_gate.npz",
    }
    payload = {
        "format": "strict-track2-v337-recursive-ood-holdout-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_holdouts": {"validation": [7, 18], "local_test": [6, 22]},
        "coverage": "all 8 alignments; exactly 512 public world-model windows",
        "label": "recursive context RGB MAE to public ground truth >=15",
        "checks_all_required": {
            "corrupt_contexts_each_split": ">=50",
            "teacher_specificity_each_split": ">=0.98",
            "recursive_negative_specificity_each_split": ">=0.93",
            "corrupt_recall_each_split": ">=0.65",
        },
        "authorizes_rl": False,
        "guards": {
            "no_reward_or_success_outcomes": True,
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
