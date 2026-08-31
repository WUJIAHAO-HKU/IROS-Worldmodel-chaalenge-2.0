#!/usr/bin/env python3
"""Preregister a bounded, reward-free recursive routing/blend sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v377_recursive_source_blend_seed1540_20260823"
RUN = JOINT / NAME
REGISTRY = OFFICIAL / "run_registry" / NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if RUN.exists() or REGISTRY.exists():
        raise FileExistsError("refusing overwrite")
    v376 = JOINT / "v376_ood_routed_cartesian_phase_seed1539_20260823/audit/recursive_reward_causal.json"
    v375_release = JOINT / "v375_bounded_cartesian_phase_pilot_seed1538_20260823/release"
    gate = JOINT / "v339_high_specificity_recursive_ood_gate_seed1508_20260822/recursive_ood_gate.npz"
    script = ROOT / "pipeline/scripts/audit_v377_recursive_source_blend_sweep.py"
    library = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for path in (v376, v375_release, gate, script, library):
        if not path.exists():
            raise FileNotFoundError(path)
    report = json.loads(v376.read_text())
    if report.get("passed") is not False:
        raise RuntimeError("v377 expects the recorded v376 rejection")
    if not all(report["teacher_frame_bit_exact"].values()):
        raise RuntimeError("v376 did not establish exact clean-context preservation")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REGISTRY.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v377-recursive-source-blend-sweep-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "replace late severe-corruption routing with stable generated-context routing and bounded correction strength",
        "calibration_episodes": [12, 33],
        "episode_disjoint_test_episodes": [20, 47],
        "configs": [
            {"threshold": threshold, "alpha": alpha}
            for threshold in (0.05, 0.10)
            for alpha in (0.25, 0.50, 0.75)
        ],
        "selection": {
            "inputs": "public train RGB/actions only; no reward or outcomes",
            "eligibility": "zero teacher routes, recursive RGB ratio <=0.90, temporal ratio <=1.03",
            "score": "minimize max(rgb_ratio/0.80, temporal_ratio/1.02)",
            "tie_break": "lower RGB ratio, lower alpha, higher threshold",
            "test_unread_until_selection": True,
        },
        "test_acceptance": {
            "zero_teacher_routes": True,
            "recursive_rgb_ratio_le_0p90": True,
            "recursive_temporal_ratio_le_1p03": True,
            "route_rate_ge_0p50_after_first_chunk": True,
            "authorizes_only_frozen_candidate_packaging_not_rl": True,
        },
        "evidence_sha256": {
            "v376_rejection": sha256(v376), "v339_gate": sha256(gate),
            "v375_cartesian_manifest": sha256(v375_release / "cartesian_phase_manifest.json"),
            "library": sha256(library), "sweep_script": sha256(script),
        },
        "guards": {
            "public_train_only": True, "reward_or_success_outcomes_used": False,
            "policy_modified": False, "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "preregistration.json").write_text(text)
    (REGISTRY / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "configs": payload["configs"]}, indent=2))


if __name__ == "__main__":
    main()
