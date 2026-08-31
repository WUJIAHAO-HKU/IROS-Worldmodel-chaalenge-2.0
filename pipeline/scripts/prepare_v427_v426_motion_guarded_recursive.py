#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v427_v426_motion_guarded_recursive_seed1577_20260823"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    candidate = JOINT / "v426_v423_motion_guarded_right_release/motion_guarded_right_manifest.json"
    manifest = json.loads(candidate.read_text())
    if float(manifest["right_normalized_motion_max"]) != 0.03:
        raise RuntimeError("v426 threshold is not the fixed 0.03 boundary")
    sources = {
        "audit": ROOT / "pipeline/scripts/audit_v427_v426_motion_guarded_recursive.py",
        "runtime": ROOT / "pipeline/wam_pipeline/v426_motion_guarded_right_runtime.py",
        "baseline_manifest": JOINT / "v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "candidate_manifest": candidate,
    }
    payload = {
        "format": "strict-track2-v427-v426-motion-guarded-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "v425 localized v423 temporal regression to higher normalized action motion while low/moderate-motion "
            "requests improved temporal, RGB, and reward error. Use the already-defined 0.03 training high-motion "
            "boundary: learned v423 at or below it and frozen v208 above it."
        ),
        "fixed_route": {
            "right_normalized_motion_max": 0.03,
            "learned_expert_when": "right motion <= 0.03",
            "frozen_expert_when": "right motion > 0.03",
            "threshold_sweep_deployed": False,
            "routing_inputs": ["instruction", "history_actions", "future_actions"],
        },
        "coverage": "same frozen validation/local public right episodes; all 512 recursively coherent windows",
        "checks": "both splits: teacher+recursive RGB <=0.99, temporal <=1.00, recursive reward error <=1.02; all required",
        "authorizes": "full-trajectory reward trace only",
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "outcomes_or_success_labels_read": False,
            "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True,
            "no_real_submission": True,
            "no_rl_authorized_by_this_gate": True,
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
