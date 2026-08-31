#!/usr/bin/env python3
"""Freeze the unique minimum-strength v416 reward-monotone candidate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v417_reward_monotone_successor_seed1571_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME
PARENT = J / "v409_half_contracted_progressive_seed1567_20260823/release"
DISCOVERY = O / "run_registry/v416_reward_monotone_successor_sweep_20260823/result.json"
REWARD_MAP = J / "v243b_public_clean_successor_reward_map_seed1445_20260819/audit/clean_successor_rewards.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v417_reward_monotone_successor_runtime.py",
        "v409_manifest": PARENT / "half_contracted_progressive_manifest.json",
        "discovery": DISCOVERY,
        "public_reward_map": REWARD_MAP,
        "audit": ROOT / "pipeline/scripts/audit_v417_recursive_reward_causal.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    discovery = json.loads(DISCOVERY.read_text())
    if discovery.get("format") != "strict-track2-v416-reward-monotone-successor-discovery-v1":
        raise RuntimeError("wrong discovery")
    selected = [
        row for row in discovery["candidates"]
        if row["policy"] == "max_reward_within32_ge0p9"
        and row["alpha_multiplier"] == 1.0
    ]
    if len(selected) != 1:
        raise RuntimeError("non-unique v417 candidate")
    selected = selected[0]
    if not (
        selected["route_positive_delta_fraction"] >= 0.8
        and selected["route_delta_lt_minus_0p01"] == 0
        and selected["groups_delta_spread_ge_0p05"] >= 12
        and selected["protected_high_reward_min_delta"] >= -0.01
        and selected["alpha_delta_correlation"] >= 0.5
    ):
        raise RuntimeError("v417 discovery gate failed")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    (RUN / "release").mkdir()
    REG.mkdir(parents=True)
    for source in PARENT.iterdir():
        os.symlink(source.resolve(), RUN / "release" / source.name)
    manifest = {
        "format": "strict-track2-v417-reward-monotone-successor-release-v1",
        "model_version": "track2-v417-v409-public-reward-monotone-successor",
        "v409_manifest_sha256": sha256(sources["v409_manifest"]),
        "public_reward_map_sha256": sha256(REWARD_MAP),
        "discovery_sha256": sha256(DISCOVERY),
        "target_policy": "highest mean official-public reward in same real demo episode between base+8 and base+32",
        "minimum_target_reward": 0.9,
        "max_advance_frames": 32,
        "contraction_scale_inherited": 0.5,
        "alpha_multiplier": 1.0,
        "terminal_precedence_inherited": True,
        "left_path_unchanged": True,
        "runtime_reads_reward_or_outcomes": False,
        "hidden_or_final_data": False,
    }
    (RUN / "release/reward_monotone_successor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prereg = {
        "format": "strict-track2-v417-reward-monotone-successor-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "fixed_candidate": manifest,
        "discovery_metrics": selected,
        "selection_rule": "smallest tested alpha multiplier satisfying every fixed v416 reward-signal condition",
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
            "targets_are_real_official_public_demo_frames": True,
            "reward_calibration_uses_official_public_training_signal_only": True,
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
    print(json.dumps({"run": str(RUN), "registry": str(REG), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
