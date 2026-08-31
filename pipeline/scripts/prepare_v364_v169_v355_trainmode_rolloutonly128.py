#!/usr/bin/env python3
"""Preregister the decisive v169-on-v355 training-mode rollout-only gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v364_v169_v355_trainmode_rolloutonly128_seed1497_20260822"
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
        raise FileExistsError("refusing overwrite of v364")
    paths = {
        "v355_manifest": J / "v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "v360_acceptance": J / "v360_v355_service_acceptance_seed1527_20260822/acceptance_report.json",
        "v360_routes": J / "v360_v355_service_acceptance_seed1527_20260822/route_audit.json",
        "v327_go_no_go": O / "runs/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/audit/training_go_no_go.json",
        "entrypoint": ROOT / "pipeline/scripts/trainmode_rollout_only_embodied_agent.py",
        "launcher": ROOT / "pipeline/scripts/launch_v364_v169_v355_trainmode_rolloutonly128.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v364-v169-v355-trainmode-rolloutonly128-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "decisive world-model signal gate before any policy update or checkpoint allocation",
        "world_model": "frozen v355",
        "policy": "frozen v169 step5 champion",
        "protocol": {"trajectories": 128, "total_envs": 32, "group_size": 4, "rollout_epochs": 4, "episode_steps": 200, "actor_seed": 1497, "environment_seed": 0, "policy_updates": 0, "checkpoint_writes": 0, "sampling_mode": "train flow_sde"},
        "gate": {"minimum_successes": 32, "minimum_success_rate": 0.25, "comparison_v327_v326": "14/128", "on_fail": "reject v355 for RL", "on_pass": "allow exactly one conservative v169 update after disk authorization"},
        "evidence_sha256": {key: sha(path) for key, path in paths.items()},
        "guards": {"public_data_only": True, "hidden_or_final_data": False, "policy_modified": False, "reward_modified": False, "real_submission": False},
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
