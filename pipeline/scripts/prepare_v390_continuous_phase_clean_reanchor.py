#!/usr/bin/env python3
"""Freeze one v390 integration after v389 train-only admission passes."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v390_continuous_phase_clean_reanchor_seed1551_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    base = J / "v385_native_batch_clean_reanchor_seed1548_20260823/release"
    phase_run = J / "v389r1_public_recursive_phase_classifier_seed1550_20260823"
    phase_gate = phase_run / "recursive_phase_gate.npz"
    phase_report = phase_run / "training_report.json"
    runtime = ROOT / "pipeline/wam_pipeline/v390_continuous_phase_clean_reanchor_runtime.py"
    feature_contract = ROOT / "pipeline/wam_pipeline/v389_public_recursive_phase_gate.py"
    backends = ROOT / "pipeline/wam_pipeline/backends.py"
    for path in (base, phase_gate, phase_report, runtime, feature_contract, backends):
        if not path.exists():
            raise FileNotFoundError(path)
    training = json.loads(phase_report.read_text())
    if not training.get("passed") or training.get("selection") is None:
        raise RuntimeError("v389 train-only gate did not pass")

    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    release = RUN / "release"
    release.mkdir()
    for name in (
        "left_expert",
        "right_expert",
        "arm_routed_autoregressive_manifest.json",
        "source_gate.npz",
        "native_batch_clean_reanchor_manifest.json",
    ):
        source = base / name
        os.symlink(source.resolve(), release / name, target_is_directory=source.is_dir())
    os.symlink(phase_gate.resolve(), release / "recursive_phase_gate.npz")
    manifest = {
        "format": "strict-track2-v390-continuous-phase-clean-reanchor-release-v1",
        "model_version": "track2-v390-v385-continuous-visual-action-phase",
        "phase_gate": "recursive_phase_gate.npz",
        "phase_gate_sha256": sha256(phase_gate),
        "phase_threshold": float(training["selection"]["threshold"]),
        "change": "replace only v385 hard action-nearest onset boolean with the frozen v389 public-train continuous RGB/action phase probability",
        "unchanged_routes": "right/source/post-grasp/action>=0.99/no-failure and endpoint clean renderer remain v385-exact",
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (release / "continuous_phase_clean_reanchor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v390-continuous-phase-clean-reanchor-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "release": str(release),
        "fixed_candidate": manifest,
        "no_parameter_sweep": True,
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
        "next_authority": "only a passing audit authorizes one 32-trajectory zero-update preflight; never direct RL",
        "evidence_sha256": {
            "v389_training_report": sha256(phase_report),
            "v389_phase_gate": sha256(phase_gate),
            "runtime": sha256(runtime),
            "feature_contract": sha256(feature_contract),
            "backends": sha256(backends),
        },
        "guards": {
            "public_world_model_data_only": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False,
            "official_reward_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "release": str(release)}, indent=2))


if __name__ == "__main__":
    main()
