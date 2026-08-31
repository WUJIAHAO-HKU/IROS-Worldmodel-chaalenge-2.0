#!/usr/bin/env python3
"""Preregister public-only v323 phase gate training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v323_public_terminal_phase_gate_seed1493_20260822"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = JOINT / NAME
    registry = OFFICIAL / "run_registry" / NAME
    if run.exists() or registry.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "trainer": ROOT / "pipeline/scripts/build_v323_public_terminal_phase_gate.py",
        "library_index": JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz",
        "instruction_map": JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "reward_checkpoint": ROOT / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt",
        "rejected_v322": JOINT / "v322_v317_causal_alpha_terminal_gate_seed1492_20260822/audit/causal_gate_report.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["rejected_v322"].read_text())["passed"] is not False:
        raise RuntimeError("v322 was not rejected")
    payload = {
        "format": "strict-track2-v323-public-terminal-phase-gate-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "learn, from public train40 right demonstrations only, when each frozen retrieval episode reaches a sustained successful terminal phase and which terminal frames actually score at least 0.90",
        "fixed_training": {
            "reward_threshold": 0.90,
            "sustained_consecutive_rows": 3,
            "phase_onset": "first public row with three consecutive terminal rewards >=0.90",
            "terminal_eligibility": "phase onset exists and frozen terminal-row reward >=0.90",
            "minimum_eligible_episodes": 12,
        },
        "runtime_boundary": "runtime may read only the frozen episode/onset/eligibility table; it may not load or call the reward model",
        "on_fail": "do not implement or evaluate v324",
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "guards": {
            "participant_component": "world-model training artifact only",
            "public_train40_only": True,
            "policy_or_simulator_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (run / "audit").mkdir(parents=True)
    registry.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (run / "preregistration.json").write_text(text)
    (registry / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(run), "registry": str(registry)}, indent=2))


if __name__ == "__main__":
    main()
