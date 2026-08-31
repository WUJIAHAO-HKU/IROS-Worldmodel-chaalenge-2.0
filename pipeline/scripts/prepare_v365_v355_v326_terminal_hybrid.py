#!/usr/bin/env python3
"""Preregister v355 dynamics under the frozen v326 terminal repair stack."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = J / "v365_v355_v326_terminal_hybrid_seed1529_20260822"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists():
        raise FileExistsError("refusing overwrite of v365")
    paths = {
        "v355_manifest": J / "v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "v326_registration": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/release_registration.json",
        "v326_runtime": ROOT / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "v326_parent_runtime": ROOT / "pipeline/wam_pipeline/v324_phase_guarded_terminal_runtime.py",
        "backend_factory": ROOT / "pipeline/wam_pipeline/backends.py",
        "contract": ROOT / "pipeline/scripts/test_v326_blended_phase_terminal.py",
        "audit": ROOT / "pipeline/scripts/audit_v326_blended_phase_terminal_gate.py",
        "v364_failure": ROOT / "artifacts/strict_track2_official_20260810/runs/v364_v169_v355_trainmode_rolloutonly128_seed1497_20260822/audit/trainmode_rollout_go_no_go.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    RUN.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v365-v355-v326-terminal-hybrid-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v365-v355-v326-terminal-hybrid",
            "parametric_parent": "frozen v355 (v202 left, v354 step300 right)",
            "terminal_stack": "frozen v326 action/phase gates and alpha=0.90 public terminal repair",
            "implementation": "existing v326 backend instantiated with the v355 arm-routed checkpoint directory",
            "new_trained_parameters": 0,
            "runtime_reward_or_outcome_access": False,
        },
        "sequence": ["numeric contract", "public recursive/reward causal audit", "service acceptance", "training-mode rollout-only128"],
        "fixed_gate_policy": "stop immediately at first failed stage; no threshold edits after observation",
        "evidence_sha256": {key: sha(path) for key, path in paths.items()},
        "guards": {"public_data_only": True, "participant_component": "world-model RGB predictor only", "policy_modified": False, "hidden_or_final_data": False, "real_submission": False},
    }
    (RUN / "release_registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(RUN)


if __name__ == "__main__":
    main()
