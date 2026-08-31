#!/usr/bin/env python3
"""Preregister the matched v326 control for the v361 rollout-only screen."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v362_v169_v326_rolloutonly32_control_seed1528_20260822"
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
        raise FileExistsError("refusing overwrite of v362")
    evidence = {
        "v361_preregistration": O / "run_registry/v361_v169_v355_rolloutonly32_seed1528_20260822/preregistration.json",
        "v361_result": O / "runs/v361_v169_v355_rolloutonly32_seed1528_20260822/audit/rolloutonly_go_no_go.json",
        "v326_manifest": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/release_registration.json",
        "v326_service_acceptance": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/service_acceptance.json",
        "v169_policy": V169,
        "launcher": ROOT / "pipeline/scripts/launch_v362_v169_v326_rolloutonly32_control.sh",
        "service_wrapper": ROOT / "pipeline/scripts/restart_v326_services.sh",
    }
    missing = [str(path) for path in evidence.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v362-v169-v326-rolloutonly32-control-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "matched control distinguishing v355 signal collapse from eval-only entrypoint effects",
        "world_model": "frozen v326 blended phase-terminal parent",
        "policy": "same frozen v169 checkpoint as v361",
        "protocol": {
            "entrypoint": "official RLinf EmbodiedEvalRunner",
            "environment": "public-data world-model HTTP environment; no hidden simulator",
            "trajectories": 32,
            "episode_steps": 200,
            "rollout_epochs": 1,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "seed": 0,
            "matched_to_v361": True,
        },
        "gate": {
            "minimum_successes": 1,
            "minimum_success_rate": 0.03125,
            "interpretation": "any success favors a v355 terminal-signal regression; zero leaves eval-path/variance ambiguity",
        },
        "evidence_sha256": {key: sha(path) for key, path in evidence.items()},
        "guards": {"public_data_only": True, "hidden_or_final_data": False, "policy_modified": False, "real_submission": False},
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
