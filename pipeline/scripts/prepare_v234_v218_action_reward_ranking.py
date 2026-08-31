#!/usr/bin/env python3
"""Preregister the public-capture action/reward ranking audit for v218."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = BASE / "artifacts/strict_track2_official_20260810"
NAME = "v234_v218_public_action_reward_ranking_20260818"
REG = OFF / "run_registry" / NAME
CAPTURE = OFF / "run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"
SCRIPT = BASE / "pipeline/scripts/replay_action_reward_alignment.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists():
        raise SystemExit(f"refusing to overwrite {REG}")
    captures = sorted(CAPTURE.glob("rollout_*.npz"))
    if len(captures) != 50:
        raise ValueError(f"expected 50 public captures, found {len(captures)}")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v234-v218-action-reward-ranking-preregistration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "test whether v218 assigns within-group reward rankings to unchanged "
            "public policy actions, rather than merely rendering reward-positive frames"
        ),
        "service_model_version": "track2-v218-public-knn-blend-alpha070-route-aware",
        "capture": {
            "path": str(CAPTURE),
            "files": len(captures),
            "first_sha256": sha256(captures[0]),
            "last_sha256": sha256(captures[-1]),
            "provenance": "public v211 pre-update rollouts only",
        },
        "fixed_metrics": [
            "within-GRPO-group terminal reward standard deviation",
            "right-gripper close versus terminal reward correlation",
            "right-joint motion versus terminal reward correlation",
            "per-instruction terminal reward distribution",
        ],
        "implementation_sha256": sha256(SCRIPT),
        "guards": {
            "policy_modified": False,
            "world_model_modified": False,
            "selection_use": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "captures": len(captures)}, indent=2))


if __name__ == "__main__":
    main()
