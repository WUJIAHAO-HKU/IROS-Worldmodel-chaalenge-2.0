#!/usr/bin/env python3
"""Preregister corrected numerical capture and recursive gates for frozen v295."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v299_v295_corrected_numeric_long_gate_seed1481_20260821"


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
    pixel_audit = O / "diagnostics/v296_v295_terminal_frame_equivariance_20260821.json"
    failed_gate = J / "v297_v295_terminal_frame_gates_seed1480_20260821/audit"
    capture_report = failed_gate / "post_grasp_reward_report.json"
    capture_details = failed_gate / "post_grasp_reward_details.npz"
    audit = json.loads(pixel_audit.read_text())
    for split, rgb_max, moving_max in (
        ("validation", -0.40, -0.30),
        ("local_test", -0.35, -0.25),
    ):
        summary = audit["summaries"][split]
        if summary["relative_mae_change"] > rgb_max:
            raise RuntimeError(f"RGB evidence failed on {split}")
        if summary["relative_moving_mae_change"] > moving_max:
            raise RuntimeError(f"moving evidence failed on {split}")
    sources = [
        B / "pipeline/wam_pipeline/v295_terminal_frame_preserving_mirror_runtime.py",
        B / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py",
        B / "pipeline/wam_pipeline/backends.py",
        B / "pipeline/scripts/restart_v295_services.sh",
        B / "pipeline/scripts/launch_v299_v295_long_gate.sh",
        B / "pipeline/scripts/verify_v298_capture_relative.py",
    ]
    registry.mkdir(parents=True)
    (run / "audit").mkdir(parents=True)
    (run / "local_dev_token.txt").write_text("local-dev-token\n")
    input_gate = run / "audit/capture_relative_gate.json"
    payload = {
        "format": "strict-track2-v299-v295-corrected-gates-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "fixed_model": {
            "model_version": "track2-v295-terminal-frame-preserving-mirror-v271",
            "parent": "track2-v271-endpoint-calibrated-terminal",
            "right_closed_frames_1_to_7": "mirror-equivariant v271",
            "frame_8": "bit-exact direct v271 RGB",
            "left_or_open_requests": "bit-exact direct v271",
        },
        "corrected_numeric_capture_gate": {
            "success_set_identical": True,
            "success_reward_max_absolute_change": 1e-4,
            "low_reward_max_absolute_change": 1e-4,
            "all_reward_mean_absolute_change": 5e-6,
            "reason": "independent CUDA reward forwards are not bitwise deterministic",
        },
        "fixed_audit": {
            "recursive_chunks": 16,
            "threshold": 0.9,
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.20,
            "margin_min": 0.55,
        },
        "input_capture_gate": str(input_gate),
        "fixed_inputs": {
            "capture_report": str(capture_report),
            "capture_report_sha256": sha(capture_report),
            "capture_details": str(capture_details),
            "capture_details_sha256": sha(capture_details),
            "pixel_audit": str(pixel_audit),
            "pixel_audit_sha256": sha(pixel_audit),
        },
        "implementation": {str(path): sha(path) for path in sources},
        "guards": {
            "participant_component": "world-model RGB service only",
            "policy_modified": False,
            "optimizer_modified": False,
            "runtime_uses_reward": False,
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
