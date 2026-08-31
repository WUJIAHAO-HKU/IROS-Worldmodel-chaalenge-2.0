#!/usr/bin/env python3
"""Preregister the unique zero-update v394 policy-distribution preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v395_v169_v394_route_trace32_seed1497_20260823"
REG = O / "run_registry" / NAME
RUN = O / "runs" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if REG.exists() or RUN.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "v394_fidelity": J / "v394_failure_calibrated_clean_reanchor_seed1555_20260823/audit/recursive_reward_causal.json",
        "v394_release_manifest": J / "v394_failure_calibrated_clean_reanchor_seed1555_20260823/release/continuous_phase_clean_reanchor_manifest.json",
        "v393r1_training": J / "v393r1_public_failure_calibrated_phase_seed1554_20260823/training_report.json",
        "runtime": ROOT / "pipeline/wam_pipeline/v391_v390_route_trace_runtime.py",
        "backend": ROOT / "pipeline/wam_pipeline/backends.py",
        "service": ROOT / "pipeline/scripts/restart_v395_trace_services.sh",
        "launcher": ROOT / "pipeline/scripts/launch_v395_v169_v394_route_trace32.sh",
        "analyzer": ROOT / "pipeline/scripts/analyze_v395_route_trace.py",
        "frozen_v169": O / "runs/v169_parent_step5_seed1243_20260813/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not json.loads(paths["v394_fidelity"].read_text()).get("passed"):
        raise RuntimeError("v394 fidelity gate did not pass")
    if not json.loads(paths["v393r1_training"].read_text()).get("passed"):
        raise RuntimeError("v393r1 training gate did not pass")
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v395-v169-v394-route-trace32-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(RUN),
        "purpose": "one-shot zero-update check that frozen v394 transfers to the frozen v169 policy distribution before any RL cost",
        "protocol": {
            "trajectories": 32,
            "rollout_epochs": 1,
            "episode_steps": 200,
            "actor_seed": 1497,
            "environment_seed": 0,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "world_model_outputs": "bit-equivalent v394; telemetry side effect only",
        },
        "gate": {
            "minimum_successes": 13,
            "rationale": "13/32 is the nearest integer non-regression threshold to the frozen historical 51/128 baseline",
            "minimum_continuous_routes": 4,
            "continuous_routes_must_exceed_hard_phase_routes": True,
            "all_required": True,
        },
        "on_pass": "authorize exactly one conservative RL update; no public batch16 or final evaluation yet",
        "on_fail": "reject v394 before RL; no outcome-based threshold tuning",
        "evidence_sha256": {name: sha256(path) for name, path in paths.items()},
        "guards": {
            "public_world_model_rollout_only": True,
            "policy_modified": False,
            "reward_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (REG / "preregistration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"registry": str(REG), "run": str(RUN)}, indent=2))


if __name__ == "__main__":
    main()
