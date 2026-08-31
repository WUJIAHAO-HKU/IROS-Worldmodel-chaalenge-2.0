#!/usr/bin/env python3
"""Pre-register the v350 public residual bridge fit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v350_public_residual_bridge_fit_seed1519_20260822"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v350_learned_residual_bridge_runtime.py",
        "fit": ROOT / "pipeline/scripts/fit_v350_public_residual_bridge_profile.py",
        "shared_collector": ROOT / "pipeline/scripts/fit_v346_public_temporal_residual_profile.py",
        "v349_holdout": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v349_v347_public_holdout_recursive_seed1518_20260822/holdout_recursive_gate_report.json",
    }
    payload = {
        "format": "strict-track2-v350-public-residual-bridge-fit-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "retrieval RGB motion bridge: anchor frame zero to v334 prediction and monotonically converge to public terminal target",
        "training": {
            "data": "15 declared public right-arm train episodes; all retrievals already cross-episode in diagnosed focused sample",
            "targets": "RGB and frame-difference pixels only; no rewards or outcomes",
            "folds": "episode-grouped 3-fold CV",
            "temporal_weight_grid": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            "constraints": "q0>=0.75; q0>=...>=q4; q4<=0.25",
            "selection_score": "0.30*OOF RGB ratio + 0.70*OOF temporal ratio",
        },
        "checks": "exact 1926 rows, >=100 direct, >=25/fold, OOF RGB and temporal ratios <=0.90",
        "authorizes": "one focused recursive train gate only",
        "guards": {"public_train_only": True, "outcomes_or_rewards_used": False,
                   "no_official_batch16_outcomes": True, "no_hidden_or_final_data": True,
                   "no_real_submission": True},
        "source_sha256": {name: sha(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
