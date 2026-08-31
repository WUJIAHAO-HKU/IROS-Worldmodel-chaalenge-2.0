#!/usr/bin/env python3
"""Freeze the structural action-phase correction to v409 progression."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v420_action_phase_progressive_seed1573_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT = J / "v409_half_contracted_progressive_seed1567_20260823/release"
V418 = O / "run_registry/v418_v417_teacher_gate_features_20260823.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v420_action_phase_progressive_runtime.py",
        "v409_manifest": PARENT / "half_contracted_progressive_manifest.json",
        "action_phase_diagnostic": V418,
        "audit": ROOT / "pipeline/scripts/audit_v420_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    diagnostic = json.loads(V418.read_text())
    if diagnostic.get("format") != "strict-track2-v418-v417-teacher-causal-gate-features-v1":
        raise RuntimeError("wrong action phase diagnostic")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    manifest = {
        "format": "strict-track2-v420-action-phase-progressive-release-v1",
        "model_version": "track2-v420-v409-action-phase-one-chunk-progressive",
        "v409_manifest_sha256": sha256(sources["v409_manifest"]),
        "action_phase_diagnostic_sha256": sha256(V418),
        "phase_anchor": "nearest clean normalized action sequence",
        "progress_offset_frames": 8,
        "contraction_scale_inherited": 0.5,
        "terminal_precedence_inherited": True,
        "left_path_unchanged": True,
        "parameter_sweep_used": False,
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (RUN / "release/action_phase_progressive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v420-action-phase-progressive-preregistration-v1",
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
        "next_authority": "Passing authorizes one frozen-v169 32-trajectory reward trace with zero updates.",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "official_public_actions_only": True,
            "simulator_outcomes_used": False,
            "policy_modified": False,
            "runtime_reads_reward_or_outcomes": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
