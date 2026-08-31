#!/usr/bin/env python3
"""Preregister and assemble the fixed v406 progressive integration."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v406_progressive_then_terminal_seed1564_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT = J / "v400_supported_posterior_blend_seed1560_20260823/release"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v406_progressive_then_terminal_runtime.py",
        "v245_runtime": ROOT / "pipeline/wam_pipeline/v245_clean_progressive_successor_runtime.py",
        "v400_manifest": PARENT / "supported_posterior_blend_manifest.json",
        "v405_trace": O / "run_registry/v405_v169_v400_progressive_trace32_seed1563_20260823/route_trace.jsonl",
        "v405_result": O / "run_registry/v405_v169_v400_progressive_trace32_seed1563_20260823/result.json",
        "audit": ROOT / "pipeline/scripts/audit_v406_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    manifest = {
        "format": "strict-track2-v406-progressive-then-terminal-release-v1",
        "model_version": "track2-v406-v245-progressive-v400-terminal",
        "v400_manifest_sha256": sha256(sources["v400_manifest"]),
        "progressive_formula": "frozen v245 unbound _alpha",
        "progressive_target": "nearest public expert row advanced by exactly 24 frames",
        "route": "right + generated-source + post-grasp + no frozen failure signature",
        "terminal_precedence": True,
        "left_path_unchanged": True,
        "no_parameter_sweep": True,
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    manifest_path = RUN / "release/progressive_then_terminal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    prereg = {
        "format": "strict-track2-v406-progressive-then-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "hypothesis": "The frozen v245 progressive potential is reachable and separates GRPO candidates, while v400 terminal precedence prevents terminal false-positive regression.",
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
        "guards": {
            "official_public_50_demo_library_only": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False,
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
