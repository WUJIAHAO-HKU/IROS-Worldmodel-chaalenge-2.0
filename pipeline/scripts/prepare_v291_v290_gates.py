#!/usr/bin/env python3
"""Preregister compliant service/reward/recursive gates for v290."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v291_v290_right_closed_mirror_gates_seed1479_20260821"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry, run = O / "run_registry" / N, J / N
    if registry.exists() or run.exists():
        raise FileExistsError("refusing overwrite")
    mirror_audit = O / "diagnostics/v289_world_model_mirror_equivariance_20260821.json"
    audit = json.loads(mirror_audit.read_text())
    for split in ("validation", "local_test"):
        summary = audit["summaries"][split]
        if summary["relative_mae_change"] > -0.35 or summary["relative_moving_mae_change"] > -0.25:
            raise RuntimeError(f"mirror evidence failed on {split}")
    sources = [
        B / "pipeline/wam_pipeline/v290_right_closed_mirror_runtime.py",
        B / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py",
        B / "pipeline/wam_pipeline/backends.py",
        B / "pipeline/scripts/test_v290_right_closed_mirror_runtime.py",
        B / "pipeline/scripts/restart_v290_services.sh",
        B / "pipeline/scripts/launch_v291_v290_gates.sh",
        B / "pipeline/scripts/audit_v289_world_model_mirror_equivariance.py",
    ]
    registry.mkdir(parents=True)
    (run / "audit").mkdir(parents=True)
    (run / "local_dev_token.txt").write_text("local-dev-token\n")
    payload = {
        "format": "strict-track2-v291-v290-gates-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "fixed_model": {
            "model_version": "track2-v290-right-closed-mirror-v271",
            "parent": "track2-v271-endpoint-calibrated-terminal",
            "right_route": "future joint motion; gripper tie-break for static terminal windows",
            "mirror_gate": "right route and mean requested right gripper <= 0.5",
            "mirror_sign": [-1, 1, 1, 1, -1, -1, 1],
            "left_requests": "bit-exact v271 path",
            "response": "RGB frames only",
        },
        "fixed_gate": {
            "capture_success_like_min": 4,
            "capture_group_std_min": 0.005,
            "capture_alignment_global_min": 0.05,
            "capture_alignment_group_min": 0.40,
            "recursive_chunks": 16,
            "recursive_threshold": 0.9,
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.20,
            "margin_min": 0.55,
            "service_contract": True,
        },
        "evidence": {
            "mirror_audit": str(mirror_audit),
            "mirror_audit_sha256": sha(mirror_audit),
            "validation_right_direct_mae": audit["summaries"]["validation"]["direct_mae"],
            "validation_right_all_mirror_mae": audit["summaries"]["validation"]["mirror_mae"],
            "local_right_direct_mae": audit["summaries"]["local_test"]["direct_mae"],
            "local_right_all_mirror_mae": audit["summaries"]["local_test"]["mirror_mae"],
            "route_accuracy_complete_heldout_windows": "1349/1349",
        },
        "implementation": {str(path): sha(path) for path in sources},
        "guards": {
            "participant_component": "world-model RGB service only",
            "policy_modified": False,
            "optimizer_modified": False,
            "returns_actions_rewards_terminations_or_success": False,
            "public_data_only": True,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (registry / "preregistration.json").write_text(text)
    (run / "release_registration.json").write_text(text)


if __name__ == "__main__":
    main()
