#!/usr/bin/env python3
"""Preregister a zero-update 32-trajectory v169-on-v355 rollout preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v361_v169_v355_rolloutonly32_seed1528_20260822"
REG = O / "run_registry" / NAME
RUN = O / "runs" / NAME
V169 = O / "runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite of v361")
    evidence = {
        "candidate_manifest": J / "v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "recursive_gate": J / "v356_v355_parametric_recursive_seed1524_20260822/recursive_gate_report.json",
        "service_acceptance": J / "v360_v355_service_acceptance_seed1527_20260822/acceptance_report.json",
        "route_audit": J / "v360_v355_service_acceptance_seed1527_20260822/route_audit.json",
        "v327_go_no_go": O / "runs/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/audit/training_go_no_go.json",
        "v169_policy": V169,
        "eval_entrypoint": ROOT / "third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/examples/embodiment/eval_embodied_agent.py",
        "launcher": ROOT / "pipeline/scripts/launch_v361_v169_v355_rolloutonly32.sh",
        "service_wrapper": ROOT / "pipeline/scripts/restart_v355_services.sh",
    }
    missing = [str(path) for path in evidence.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    for key in ("service_acceptance", "route_audit"):
        if json.loads(evidence[key].read_text()).get("passed") is not True:
            raise RuntimeError(f"failed prerequisite: {key}")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v361-v169-v355-rolloutonly32-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "cheap zero-update signal screen before any expensive RL checkpoint",
        "world_model": "frozen v355 v202-left/v354-right parametric parent",
        "policy": "frozen historical v169 step5 champion (51/128)",
        "protocol": {
            "entrypoint": "official RLinf EmbodiedEvalRunner",
            "environment": "public-data world-model HTTP environment wired as eval; not the official hidden simulator",
            "trajectories": 32,
            "episode_steps": 200,
            "rollout_epochs": 1,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "seed": 0,
        },
        "gate": {
            "minimum_successes": 8,
            "minimum_success_rate": 0.25,
            "baseline_v327_initial_rollout": "14/128 = 0.109375",
            "on_fail": "reject v355 for RL before training",
            "on_pass": "permit preregistration of exactly one conservative RL update",
        },
        "evidence_sha256": {key: sha(path) for key, path in evidence.items()},
        "guards": {
            "public_data_only": True,
            "hidden_or_final_data": False,
            "policy_modified": False,
            "reward_modified": False,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
