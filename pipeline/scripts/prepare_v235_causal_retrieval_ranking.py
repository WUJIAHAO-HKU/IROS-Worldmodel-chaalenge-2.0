#!/usr/bin/env python3
"""Preregister the v235 public causal-retrieval ranking sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v235_public_causal_retrieval_ranking_seed1434_20260818"
REG = OFF / "run_registry" / NAME
CAPTURE = OFF / "run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"
LIBRARY = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
SCRIPT = BASE / "pipeline/scripts/sweep_v235_causal_retrieval_ranking.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists():
        raise SystemExit(f"refusing to overwrite {REG}")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v235-causal-retrieval-ranking-preregistration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "find a physical right-arm world-model retrieval gate with correct action ranking",
        "inputs": {
            "capture": str(CAPTURE),
            "capture_files": len(list(CAPTURE.glob("rollout_*.npz"))),
            "library": str(LIBRARY),
            "library_sha256": sha256(LIBRARY),
        },
        "fixed_candidates": [
            "parent",
            "fixed070_raw",
            "causal070_raw",
            "causal_soft070_raw",
            "causal_hard070_raw",
            "fixed070_delta",
            "causal070_delta",
            "causal_soft070_delta",
            "causal_soft100_delta",
            "causal_hard070_delta",
        ],
        "selection_rule": (
            "maximize right expert-similarity/reward correlation; require nonzero "
            "within-group reward variance and no worse motion correlation than parent"
        ),
        "implementation_sha256": sha256(SCRIPT),
        "guards": {
            "policy_modified": False,
            "public_data_only": True,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
