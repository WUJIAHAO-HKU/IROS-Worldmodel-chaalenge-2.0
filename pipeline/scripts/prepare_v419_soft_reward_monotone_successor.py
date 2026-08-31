#!/usr/bin/env python3
"""Freeze the low-strength fallback declared by the v417 failure mode."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v419_soft_reward_monotone_successor_seed1572_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT = J / "v417_reward_monotone_successor_seed1571_20260823/release"
DISCOVERY = O / "run_registry/v416_reward_monotone_successor_sweep_20260823/result.json"
V417_AUDIT = O / "run_registry/v417_reward_monotone_successor_seed1571_20260823/recursive_reward_causal.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v419_soft_reward_monotone_successor_runtime.py",
        "v417_manifest": PARENT / "reward_monotone_successor_manifest.json",
        "discovery": DISCOVERY,
        "v417_audit": V417_AUDIT,
        "audit": ROOT / "pipeline/scripts/audit_v419_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    v417 = json.loads(V417_AUDIT.read_text())
    if v417.get("passed") is not False:
        raise RuntimeError("v419 requires rejected v417")
    failed = v417["checks"]
    if failed.get("validation_false_positive_rate_nonregression") is not False:
        raise RuntimeError("v417 failure was not premature success")
    discovery = json.loads(DISCOVERY.read_text())
    selected = [
        row for row in discovery["candidates"]
        if row["policy"] == "max_reward_within32_ge0p9"
        and row["alpha_multiplier"] == 0.5
    ]
    if len(selected) != 1:
        raise RuntimeError("non-unique v419 fallback")
    selected = selected[0]
    if not (
        selected["route_positive_delta_fraction"] >= 0.8
        and selected["route_delta_lt_minus_0p01"] == 0
        and selected["groups_reward_spread_ge_0p05"] >= 20
        and selected["candidate_hits_ge_0p9"] - selected["v400_hits_ge_0p9"] <= 4
        and selected["protected_high_reward_min_delta"] >= -0.01
    ):
        raise RuntimeError("v419 discovery fallback gate failed")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    manifest = {
        "format": "strict-track2-v419-soft-reward-monotone-successor-release-v1",
        "model_version": "track2-v419-v417-soft-public-reward-progress",
        "v417_manifest_sha256": sha256(sources["v417_manifest"]),
        "discovery_sha256": sha256(DISCOVERY),
        "effective_alpha_scale": 0.25,
        "v416_alpha_multiplier": 0.5,
        "target_policy_inherited": "max_reward_within32_ge0p9",
        "purpose": "continuous progress reward below premature binary-success regime",
        "runtime_reads_reward_or_outcomes": False,
        "hidden_or_final_data": False,
    }
    (RUN / "release/soft_reward_monotone_successor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v419-soft-reward-monotone-successor-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "fixed_candidate": manifest,
        "discovery_metrics": selected,
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
        "next_authority": "Passing authorizes one fresh-seed frozen-v169 32-trajectory reward trace only, with zero policy updates.",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "official_public_data_and_training_reward_only": True,
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
    print(json.dumps({"run": str(RUN), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
