#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v386_v169_v385_trainmode_rolloutonly128_seed1497_20260823"
REG = O / "run_registry" / NAME
RUN = O / "runs" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "v385_registration": J / "v385_native_batch_clean_reanchor_seed1548_20260823/release_registration.json",
        "v385_recursive_gate": J / "v385_native_batch_clean_reanchor_seed1548_20260823/audit/recursive_reward_causal.json",
        "v385_service_acceptance": J / "v385_native_batch_clean_reanchor_seed1548_20260823/audit/service_acceptance.json",
        "v169_policy": O / "runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt",
        "entrypoint": ROOT / "pipeline/scripts/trainmode_rollout_only_embodied_agent.py",
        "launcher": ROOT / "pipeline/scripts/launch_v386_v169_v385_rolloutonly128.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    for key in ("v385_recursive_gate", "v385_service_acceptance"):
        if json.loads(paths[key].read_text()).get("passed") is not True:
            raise RuntimeError(f"failed prerequisite: {key}")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v386-v169-v385-rolloutonly128-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "decisive policy-distribution world-model signal gate before any further policy update",
        "world_model": "frozen v385 native-batch action-phase clean reanchor",
        "policy": "frozen historical v169 step5 champion (51/128)",
        "protocol": {
            "trajectories": 128, "total_envs": 32, "group_size": 4,
            "rollout_epochs": 4, "episode_steps": 200, "actor_seed": 1497,
            "environment_seed": 0, "policy_updates": 0, "checkpoint_writes": 0,
            "sampling_mode": "official train flow_sde",
        },
        "gate": {
            "minimum_successes": 32, "minimum_success_rate": 0.25,
            "comparators": {"v327_v326": "14/128", "v383_v382": "8/128"},
            "on_fail": "reject v385 before RL training",
            "on_pass": "permit preregistration of exactly one conservative v169 update",
        },
        "evidence_sha256": {key: sha256(path) for key, path in paths.items()},
        "guards": {
            "public_data_only": True, "hidden_or_final_data": False,
            "policy_modified": False, "reward_modified": False, "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
