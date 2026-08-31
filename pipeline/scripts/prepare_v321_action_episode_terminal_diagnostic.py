#!/usr/bin/env python3
"""Preregister the v321 public-only terminal lookup diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFFICIAL = ROOT / "artifacts/strict_track2_official_20260810"
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
NAME = "v321_action_episode_terminal_diagnostic_seed1491_20260822"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = JOINT / NAME
    registry = OFFICIAL / "run_registry" / NAME
    if run.exists() or registry.exists():
        raise FileExistsError("refusing overwrite")
    paths = {
        "diagnostic": ROOT / "pipeline/scripts/diagnose_v321_action_episode_terminal.py",
        "parent_runtime": ROOT / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py",
        "library_index": JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz",
        "split": ROOT / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": JOINT / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "reward_checkpoint": ROOT / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "format": "strict-track2-v321-action-episode-terminal-diagnostic-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "v271 selects the right terminal episode from context alone; using the already-computed public action+visual nearest row to choose the terminal episode should repair the episode22 right-success blind spot without changing actions, reward, or policy",
        "variants": {
            "baseline": "terminal episode from context-only nearest public row",
            "candidate": "terminal episode from action+visual nearest public row; all other v271 math unchanged",
        },
        "data_boundary": {
            "decision": "public validation right episodes 7 and 18",
            "confirmation_only": "episode-disjoint public local-test right episodes 6 and 22",
            "hidden_or_final": False,
            "simulator_outcomes": False,
            "policy_checkpoint": False,
        },
        "fixed_checks": {
            "validation_transition_count_ge_12": True,
            "confirmation_transition_count_ge_12": True,
            "validation_reward_mae_ratio_le_0p95": True,
            "validation_rgb_mae_ratio_le_1p05": True,
            "validation_transition_mean_improvement_ge_minus_0p02": True,
            "confirmation_reward_mae_nonregression": True,
            "confirmation_transition_mean_improvement_ge_0p20": True,
            "confirmation_transition_hit_rate_ge_0p80": True,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "on_pass": "implement a new world-model runtime, then run contract/counterfactual/left-nonregression gates before any RL",
        "on_fail": "reject this lookup change and do not launch RL",
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "guards": {
            "participant_component": "world-model diagnostic only",
            "official_policy_reward_or_rl_changed": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    (run / "audit").mkdir(parents=True)
    registry.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (run / "preregistration.json").write_text(text)
    (registry / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(run), "registry": str(registry)}, indent=2))


if __name__ == "__main__":
    main()
