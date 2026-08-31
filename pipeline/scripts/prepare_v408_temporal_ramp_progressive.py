#!/usr/bin/env python3
"""Preregister the temporal-continuity correction to frozen v407."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v408_temporal_ramp_progressive_seed1566_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT_RUN = J / "v407_one_chunk_progressive_seed1565_20260823"
PARENT = PARENT_RUN / "release"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v408_temporal_ramp_progressive_runtime.py",
        "v407_manifest": PARENT / "one_chunk_progressive_manifest.json",
        "v407_audit": PARENT_RUN / "audit/recursive_reward_causal.json",
        "audit": ROOT / "pipeline/scripts/audit_v408_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    prior = json.loads(sources["v407_audit"].read_text())
    failed = [name for name, value in prior["checks"].items() if not value]
    if failed != [
        "validation_recursive_temporal_le1p05",
        "local_recursive_temporal_le1p05",
    ]:
        raise RuntimeError(f"unexpected v407 failure set: {failed}")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    ramp = [round(index / 8.0, 3) for index in range(1, 9)]
    manifest = {
        "format": "strict-track2-v408-temporal-ramp-progressive-release-v1",
        "model_version": "track2-v408-v407-linear-temporal-ramp",
        "v407_manifest_sha256": sha256(sources["v407_manifest"]),
        "ramp_weights": ramp,
        "terminal_frame_alpha_preserved": True,
        "rationale": "repair v407's sole temporal-delta failure without changing its final predicted frame",
        "terminal_precedence_inherited": True,
        "left_path_unchanged": True,
        "no_parameter_sweep": True,
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (RUN / "release/temporal_ramp_progressive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v408-temporal-ramp-progressive-preregistration-v1",
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
        "next_authority": "Passing audit authorizes one output-changing rollout-only trace, not policy training.",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {"official_public_50_demo_library_only": True, "policy_modified": False, "runtime_reads_reward_or_outcomes": False, "hidden_or_final_data": False, "real_submission": False},
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
