#!/usr/bin/env python3
"""Pre-register the numerically normalized v347 public residual fit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v347_public_temporal_residual_fit_seed1516_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v346_learned_temporal_residual_runtime.py",
        "fit": ROOT / "pipeline/scripts/fit_v346_public_temporal_residual_profile.py",
        "failed_v346_prereg": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v346_public_temporal_residual_fit_seed1515_20260822/release_registration.json",
        "failed_v346_log": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v346_public_temporal_residual_fit_seed1515_20260822/fit.log",
    }
    payload = {
        "format": "strict-track2-v347-public-temporal-residual-fit-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "change_from_v346": "divide quadratic A and b by mean diagonal scale before constrained SLSQP; no model/data/threshold change",
        "model": "five monotone continuous residual coefficients; terminal coefficient >=0.75",
        "training": {
            "data": "all 15 declared public right-arm training episodes, 8 recursive alignments",
            "targets": "RGB next-context pixels and frame-difference pixels only",
            "folds": "episode-grouped 3-fold CV, sorted episode index modulo 3",
            "temporal_weight_grid": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            "selection_score": "0.30*OOF RGB ratio + 0.70*OOF temporal ratio",
        },
        "checks": "exact 1926 rows, >=100 direct, >=25/fold, OOF RGB and temporal ratios <=0.90",
        "authorizes": "one recursive public-train runtime gate only",
        "guards": {
            "public_train_only": True, "outcomes_or_rewards_used": False,
            "no_official_batch16_outcomes": True, "no_hidden_or_final_data": True,
            "no_real_submission": True,
        },
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
