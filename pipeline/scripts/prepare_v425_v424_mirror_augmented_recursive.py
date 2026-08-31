#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v425_v424_mirror_augmented_recursive_seed1576_20260823"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "audit": ROOT / "pipeline/scripts/audit_v425_v424_mirror_augmented_recursive.py",
        "baseline_manifest": JOINT / "v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "candidate_manifest": JOINT / "v424_v202_v423_mirror_augmented_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "v423_selection": JOINT / "v423_v354s150_mirror_augmented_right_dynamics_seed1575_20260823/selection_registration.json",
    }
    payload = {
        "format": "strict-track2-v425-v424-mirror-augmented-recursive-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coverage": "same frozen validation/local public right episodes; all 512 recursive windows",
        "checks": (
            "both splits: teacher+recursive RGB <=0.99, temporal <=1.00, recursive reward error <=1.02; all required"
        ),
        "authorizes": "full-trajectory reward trace only",
        "guards": {
            "outcomes_or_success_labels_read": False,
            "no_official_batch16_outcomes": True,
            "no_hidden_or_final_data": True,
            "no_real_submission": True,
            "no_rl_authorized_by_this_gate": True,
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
