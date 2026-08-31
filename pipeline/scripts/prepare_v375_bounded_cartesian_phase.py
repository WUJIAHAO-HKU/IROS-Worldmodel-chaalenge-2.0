#!/usr/bin/env python3
"""Freeze and preregister the v375 bounded Cartesian phase pilot."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v375_bounded_cartesian_phase_pilot_seed1538_20260823"
RUN = JOINT / NAME
REGISTRY = OFFICIAL / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REGISTRY.exists():
        raise FileExistsError("refusing overwrite")
    parent = JOINT / "v355_v202_v354_parametric_arm_routed_release"
    pose = JOINT / "v373_cartesian_pose_progress_seed1536_20260823/model/best.pt"
    pose_gate = JOINT / "v373_cartesian_pose_progress_seed1536_20260823/audit/pose_reconstruction_acceptance.json"
    phase_gate = JOINT / "v374_cartesian_phase_successor_diagnostic_seed1537_20260823/bounded_sweep_report.json"
    phase_details = JOINT / "v374_cartesian_phase_successor_diagnostic_seed1537_20260823/phase_gate_details.npz"
    runtime = ROOT / "pipeline/wam_pipeline/v375_bounded_cartesian_phase_runtime.py"
    backends = ROOT / "pipeline/wam_pipeline/backends.py"
    library = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for path in (parent, pose, pose_gate, phase_gate, phase_details, runtime, backends, library):
        if not path.exists():
            raise FileNotFoundError(path)
    if not json.loads(pose_gate.read_text())["passed"]:
        raise RuntimeError("v373 pose reconstruction was rejected")
    sweep = json.loads(phase_gate.read_text())
    selected = sweep["selected"]; tested = sweep["test_selected"]
    if (selected["radius"], selected["penalty"]) != (16, 1.0):
        raise RuntimeError("unexpected calibration selection")
    fixed_checks = {
        "test_phase_ratio_le_0p90": tested["phase_ratio"] <= 0.90,
        "test_terminal_phase_ratio_le_0p90": tested["terminal_phase_ratio"] <= 0.90,
        "test_rgb_ratio_le_0p90": tested["rgb_ratio"] <= 0.90,
        "test_delta_rgb_ratio_le_0p90": tested["delta_rgb_ratio"] <= 0.90,
    }
    if not all(fixed_checks.values()):
        raise RuntimeError(f"bounded test gate failed: {fixed_checks}")
    with np.load(phase_details, allow_pickle=False) as details:
        scale = details["scale"].astype(np.float32).tolist()

    RUN.mkdir(parents=True); (RUN / "audit").mkdir(); REGISTRY.mkdir(parents=True)
    release = RUN / "release"; release.mkdir()
    source_manifest = json.loads((parent / "arm_routed_autoregressive_manifest.json").read_text())
    os.symlink((parent / "left_expert").resolve(), release / "left_expert", target_is_directory=True)
    os.symlink((parent / "right_expert").resolve(), release / "right_expert", target_is_directory=True)
    arm_manifest = dict(source_manifest)
    arm_manifest["model_version"] = "track2-v375-parent-v202-left-v354-right"
    (release / "arm_routed_autoregressive_manifest.json").write_text(json.dumps(arm_manifest, indent=2) + "\n")
    os.symlink(pose.resolve(), release / "cartesian_pose.pt")
    cartesian_manifest = {
        "format": "strict-track2-v375-bounded-cartesian-phase-release-v1",
        "model_version": "track2-v375-bounded-cartesian-phase-r16-p1",
        "pose_checkpoint": "cartesian_pose.pt",
        "pose_checkpoint_sha256": sha256(pose),
        "radius": 16,
        "penalty": 1.0,
        "pose_scale": scale,
        "coarse_route": "visual-only nearest public train40 right trajectory window",
        "fine_route": "monotonic Cartesian pose search around each coarse phase",
        "renderer": "request last RGB plus retrieved public RGB temporal delta",
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (release / "cartesian_phase_manifest.json").write_text(json.dumps(cartesian_manifest, indent=2) + "\n")
    payload = {
        "format": "strict-track2-v375-bounded-cartesian-phase-pilot-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": cartesian_manifest["model_version"],
        "release": str(release),
        "selection": {"calibration": selected, "episode_disjoint_test": tested, "checks": fixed_checks},
        "fixed_next_gates": {
            "runtime_equivalence": "512 right holdout windows must reproduce bounded analysis within numeric tolerance",
            "http_service_acceptance": "all contract and deterministic restart tests",
            "recursive_reward_causal": (
                "frozen public right chains must preserve nonzero reward dynamics and improve over v355; "
                "all checks required before fixed official RL"
            ),
            "no_policy_training_before_all_gates": True,
        },
        "evidence_sha256": {
            str(path): sha256(path) for path in (pose, pose_gate, phase_gate, phase_details, runtime, backends, library)
        },
        "release_sha256": {
            "arm_manifest": sha256(release / "arm_routed_autoregressive_manifest.json"),
            "cartesian_manifest": sha256(release / "cartesian_phase_manifest.json"),
            "pose": sha256(pose),
        },
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "public_world_model_data_only": True,
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REGISTRY / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "release": str(release), "model_version": payload["model_version"]}, indent=2))


if __name__ == "__main__":
    main()
