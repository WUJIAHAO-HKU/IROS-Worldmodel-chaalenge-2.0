#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v239_public_motion_gated_transport_seed1438_20260819"
REG = OFF / "run_registry" / NAME
CAPTURE = OFF / "run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"
LIBRARY = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
SCRIPT = BASE / "pipeline/scripts/analyze_v239_motion_gated_transport.py"


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
        "format": "strict-track2-v239-motion-gated-transport-preregistration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "prevent post-grasp retrieval reward from preferring static right-arm actions",
        "inputs": {"capture": str(CAPTURE), "capture_files": len(list(CAPTURE.glob("rollout_*.npz"))),
                   "library": str(LIBRARY), "library_sha256": sha256(LIBRARY)},
        "fixed_library_path_quantiles": [0.50, 0.75, 0.90],
        "fixed_motion_gates": ["public_q50", "public_q75", "selected_min"],
        "selection_rule": (
            "maximize the minimum of alpha correlations with expert similarity, motion, and directional alignment; "
            "all three correlations must be positive"
        ),
        "implementation_sha256": sha256(SCRIPT),
        "guards": {"policy_modified": False, "public_data_only": True,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
