#!/usr/bin/env python3
"""Preregister a v326 rollout-only control matched to v327 sampling."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v363_v169_v326_matched_rolloutonly32_seed1497_20260822"
REG = O / "run_registry" / NAME
RUN = O / "runs" / NAME


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite of v363")
    paths = {
        "v327_preregistration": O / "run_registry/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/preregistration.json",
        "v327_go_no_go": O / "runs/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/audit/training_go_no_go.json",
        "v362_result": O / "runs/v362_v169_v326_rolloutonly32_control_seed1528_20260822/audit/rolloutonly_go_no_go.json",
        "generic_runner": ROOT / "pipeline/scripts/run_v169_worldmodel_rolloutonly32.sh",
        "launcher": ROOT / "pipeline/scripts/launch_v363_v169_v326_matched_rolloutonly32.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v363-v169-v326-matched-rolloutonly32-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "verify eval-only reproducibility after matching v327 actor seed and GRPO group structure",
        "world_model": "frozen v326",
        "policy": "frozen v169 step5",
        "protocol": {"trajectories": 32, "episode_steps": 200, "rollout_epochs": 1, "policy_updates": 0, "checkpoint_writes": 0, "actor_seed": 1497, "environment_seed": 0, "group_size": 4},
        "gate": {"minimum_successes": 1, "minimum_success_rate": 0.03125, "on_fail": "eval-only first-epoch proxy is not predictive; do not use it for v355 selection", "on_pass": "run one matched v355 paired preflight"},
        "evidence_sha256": {key: sha(path) for key, path in paths.items()},
        "guards": {"public_data_only": True, "hidden_or_final_data": False, "policy_modified": False, "real_submission": False},
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
