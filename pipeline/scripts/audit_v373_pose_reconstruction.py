#!/usr/bin/env python3
"""Verify the preregistered v373 pose reconstruction checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    output = args.run / "audit/pose_reconstruction_acceptance.json"
    if output.exists():
        raise FileExistsError("refusing overwrite")
    prereg = json.loads((args.run / "release_registration.json").read_text())
    checkpoint = args.run / "model/best.pt"
    manifest_path = args.run / "model/training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = state["dev_metrics"]
    worst = max(float(row["landmark_mae_px"]) for row in metrics["arms"].values())
    right = metrics["arms"]["arm1"]
    limits = prereg["fixed_acceptance"]
    checks = {
        "checkpoint_finite": all(torch.isfinite(value).all().item() for value in state["model"].values()),
        "exact_training_steps": int(manifest["steps"]) == int(prereg["fixed_reconstruction"]["steps"]),
        "dev_episodes_frozen": manifest["dev_episodes"] == prereg["fixed_reconstruction"]["dev_episodes"],
        "worst_arm_landmark_mae": worst <= limits["worst_arm_landmark_mae_px_max"],
        "right_origin_mae": float(right["origin_mae_px"]) <= limits["right_origin_mae_px_max"],
        "right_depth_mae": float(right["depth_mae_m"]) <= limits["right_depth_mae_m_max"],
    }
    report = {
        "format": "strict-track2-v373-pose-reconstruction-acceptance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": int(state["step"]),
        "checkpoint_sha256": sha256(checkpoint),
        "metrics": metrics,
        "worst_arm_landmark_mae_px": worst,
        "checks": checks,
        "passed": all(checks.values()),
        "guards": prereg["guards"],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
