#!/usr/bin/env python3
"""Freeze v337 train-only recursive OOD gate protocol before fitting."""

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
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "feature_contract": ROOT / "pipeline/wam_pipeline/v337_public_recursive_ood_gate.py",
        "trainer": ROOT / "pipeline/scripts/train_v337_public_recursive_ood_gate.py",
        "v326_runtime": ROOT / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "action_gate": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "phase_gate": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "split": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    payload = {
        "format": "strict-track2-v337-public-recursive-ood-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "detect bridge-recursive RGB corruption without reward or outcome labels",
        "training_rule": {
            "episodes": "exact 15 declared public-train right-arm episodes",
            "alignments": list(range(8)),
            "recursive_update": "next context equals v326 prediction[-5:]",
            "negative": "all teacher contexts plus recursive contexts with RGB MAE below 15",
            "positive": "recursive context RGB MAE to public ground truth at least 15",
            "corruption_mae_floor": 15.0,
            "feature_contract": "scene-agnostic RGB/gradient/temporal summaries only",
        },
        "selection_rule": {
            "grouping": "leave-one-public-train-episode-out",
            "c_values": [0.01, 0.1, 1.0],
            "thresholds": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99],
            "eligibility": "teacher specificity >=.99, all-negative specificity >=.95, corrupt recall >=.70",
            "tie_break": "recall, negative specificity, teacher specificity, smaller C",
        },
        "authorizes_rl": False,
        "guards": {
            "public_train_only": True,
            "reward_or_success_outcomes": False,
            "public_batch16_or_holdout_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
