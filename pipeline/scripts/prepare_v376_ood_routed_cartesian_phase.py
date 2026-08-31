#!/usr/bin/env python3
"""Freeze and preregister the v376 recursive-OOD routed Cartesian pilot."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v376_ood_routed_cartesian_phase_seed1539_20260823"
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
    v375 = JOINT / "v375_bounded_cartesian_phase_pilot_seed1538_20260823"
    parent_release = v375 / "release"
    v375_gate = v375 / "audit/recursive_reward_causal.json"
    gate_run = JOINT / "v339_high_specificity_recursive_ood_gate_seed1508_20260822"
    ood_gate = gate_run / "recursive_ood_gate.npz"
    calibration = gate_run / "calibration_report.json"
    holdout = gate_run / "holdout_report.json"
    runtime = ROOT / "pipeline/wam_pipeline/v376_ood_routed_cartesian_phase_runtime.py"
    backends = ROOT / "pipeline/wam_pipeline/backends.py"
    library = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for path in (
        parent_release, v375_gate, ood_gate, calibration, holdout, runtime, backends, library
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    v375_report = json.loads(v375_gate.read_text())
    if v375_report.get("passed") is not False:
        raise RuntimeError("v376 expects the recorded v375 mixed-context rejection")
    calibration_report = json.loads(calibration.read_text())
    holdout_report = json.loads(holdout.read_text())
    if not calibration_report.get("passed") or not holdout_report.get("passed"):
        raise RuntimeError("v339 OOD gate prerequisites did not pass")
    selected = calibration_report["selected"]
    if float(selected["threshold"]) != 0.925:
        raise RuntimeError("unexpected v339 train-selected threshold")
    if min(
        holdout_report["aggregates"][split]["teacher_specificity"]
        for split in ("validation", "local_test")
    ) < 0.98:
        raise RuntimeError("v339 teacher specificity prerequisite failed")

    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REGISTRY.mkdir(parents=True)
    release = RUN / "release"
    release.mkdir()
    for name in (
        "left_expert", "right_expert", "cartesian_pose.pt",
        "arm_routed_autoregressive_manifest.json", "cartesian_phase_manifest.json",
    ):
        os.symlink((parent_release / name).resolve(), release / name,
                   target_is_directory=(parent_release / name).is_dir())
    os.symlink(ood_gate.resolve(), release / "recursive_ood_gate.npz")
    route_manifest = {
        "format": "strict-track2-v376-ood-routed-cartesian-phase-release-v1",
        "model_version": "track2-v376-v355-default-v375-on-v339-ood",
        "recursive_ood_gate": "recursive_ood_gate.npz",
        "recursive_ood_gate_sha256": sha256(ood_gate),
        "threshold": 0.925,
        "default_route": "bit-exact frozen v355 arm-routed parametric world model",
        "corruption_route": "frozen v375 bounded Cartesian phase correction for right arm only",
        "gate_inputs": "five request RGB context frames only",
        "stateful_or_request_identity_routing": False,
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    route_manifest_path = release / "ood_routed_cartesian_manifest.json"
    route_manifest_path.write_text(json.dumps(route_manifest, indent=2) + "\n")
    checks = {
        "v339_calibration_passed": calibration_report["passed"] is True,
        "v339_holdout_passed": holdout_report["passed"] is True,
        "v339_teacher_specificity_ge_0p98_both_splits": all(
            holdout_report["aggregates"][split]["teacher_specificity"] >= 0.98
            for split in ("validation", "local_test")
        ),
        "v375_recursive_rgb_ratio_le_0p55_both_splits": all(
            v375_report["v375_over_v355"][split]["recursive_next_context_rgb_mae"] <= 0.55
            for split in ("validation", "local_test")
        ),
        "v375_recursive_reward_error_improves_both_splits": all(
            v375_report["v375_over_v355"][split]["recursive_reward_absolute_error"] < 1.0
            for split in ("validation", "local_test")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v376 design prerequisites failed: {checks}")
    payload = {
        "format": "strict-track2-v376-ood-routed-cartesian-phase-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": route_manifest["model_version"],
        "release": str(release),
        "hypothesis": (
            "v355 is preserved on clean teacher contexts; the v339 visual-only OOD gate "
            "activates v375 only after recursive image drift, combining complementary strengths"
        ),
        "design_prerequisites": checks,
        "fixed_recursive_gate": {
            "exact_rows": 512,
            "teacher_next_context_bit_exact_v355_both_splits": True,
            "teacher_reward_metrics_bit_exact_v355_both_splits": True,
            "recursive_rgb_ratio_le_0p80_both_splits": True,
            "recursive_temporal_ratio_le_1p02_both_splits": True,
            "recursive_reward_error_ratio_lt_1p00_both_splits": True,
            "reward_mean_gain_ge_0p01_or_hit_gain_ge_0p02_both_splits": True,
            "all_checks_required_before_runtime_or_rl": True,
        },
        "evidence_sha256": {
            str(path): sha256(path)
            for path in (v375_gate, ood_gate, calibration, holdout, runtime, backends, library)
        },
        "release_sha256": {
            "arm_manifest": sha256(release / "arm_routed_autoregressive_manifest.json"),
            "cartesian_manifest": sha256(release / "cartesian_phase_manifest.json"),
            "routing_manifest": sha256(route_manifest_path),
            "pose": sha256(release / "cartesian_pose.pt"),
            "ood_gate": sha256(release / "recursive_ood_gate.npz"),
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
    print(json.dumps({"run": str(RUN), "release": str(release), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
