#!/usr/bin/env python3
"""Freeze the public-train-calibrated half-contracted v407 candidate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v409_half_contracted_progressive_seed1567_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT = J / "v407_one_chunk_progressive_seed1565_20260823/release"
CALIBRATION = J / "v409_public_train_half_contraction_calibration_seed1567_20260823.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v409_half_contracted_progressive_runtime.py",
        "v407_manifest": PARENT / "one_chunk_progressive_manifest.json",
        "calibration": CALIBRATION,
        "audit": ROOT / "pipeline/scripts/audit_v409_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    calibration = json.loads(CALIBRATION.read_text())
    if calibration.get("format") != "strict-track2-v409-public-train-half-contraction-calibration-v1":
        raise RuntimeError("wrong calibration")
    if calibration["selection"].get("contraction_scale") != 0.5:
        raise RuntimeError("calibration scale drift")
    guards = calibration.get("guards", {})
    if not guards.get("public_train_windows_only") or not guards.get("validation_windows_read") is False:
        raise RuntimeError("calibration split violation")
    if guards.get("reward_or_outcomes_read") is not False:
        raise RuntimeError("calibration outcome violation")

    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    manifest = {
        "format": "strict-track2-v409-half-contracted-progressive-release-v1",
        "model_version": "track2-v409-v407-half-contracted-progressive",
        "v407_manifest_sha256": sha256(sources["v407_manifest"]),
        "calibration_sha256": sha256(CALIBRATION),
        "contraction_scale": 0.5,
        "effective_progressive_alpha_max": 0.5,
        "terminal_precedence_inherited": True,
        "left_path_unchanged": True,
        "no_reward_parameter_sweep": True,
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (RUN / "release/half_contracted_progressive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v409-half-contracted-progressive-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "fixed_candidate": manifest,
        "fixed_recursive_gate": {
            "exact_rows": 512,
            "teacher_bit_exact_v326": True,
            "recursive_rgb_ratio_max_both": 1.05,
            "recursive_temporal_ratio_max_both": 1.05,
            "recursive_reward_error_ratio_lt_both": 1.0,
            "positive_recall_min_both": 0.50,
            "positive_recall_nonregression_vs_v326": True,
            "false_positive_rate_nonregression_vs_v326": True,
            "all_required": True,
        },
        "next_authority": "Passing audit authorizes one 32-trajectory output-changing rollout preflight only.",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "official_public_data_only": True,
            "scale_selected_without_validation_reward_or_outcomes": True,
            "policy_modified": False,
            "runtime_reads_reward_or_outcomes": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
