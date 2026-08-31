#!/usr/bin/env python3
"""Preregister the fail-closed v290 full-terminal causal gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v310_v290_full_terminal_causal_gate_seed1483_20260822"
RUN = J / N
REG = O / "run_registry" / N


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing to overwrite v310")
    script = B / "pipeline/scripts/audit_v310_full_mirror_causal_gate.py"
    runtime = B / "pipeline/wam_pipeline/v290_right_closed_mirror_runtime.py"
    parent = B / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py"
    split = B / "artifacts/splits/adjust_bottle_50episodes_full.json"
    instruction_map = (
        J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
    )
    library = (
        J / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    )
    reward = B / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
    required = [script, runtime, parent, split, instruction_map, library, reward]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    payload = {
        "format": "strict-track2-v310-v290-full-terminal-causal-gate-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "reconsider the previously rejected full-terminal mirror using public "
            "ground-truth visual/reward fidelity and fixed physical counterfactuals, "
            "instead of preserving v271's self-referential high-reward predictions"
        ),
        "candidate": {
            "model_version": "track2-v290-right-closed-full-terminal-mirror-v271",
            "right_closed_path": "mirror context/actions, run frozen v271, unmirror all eight RGB frames",
            "left_or_open_path": "bit-exact frozen v271",
            "response": "eight RGB frames only",
        },
        "fixed_data": {
            "source": "declared public 50 successful demonstration episodes only",
            "selection_episodes": [7, 18],
            "confirmation_episodes": [6, 22],
            "samples_per_split": 64,
            "episode_disjoint": True,
            "simulator_outcomes_used": False,
            "hidden_or_final_data": False,
        },
        "fixed_counterfactuals": [
            "open_gripper",
            "static_transport",
            "reverse_transport",
        ],
        "fixed_thresholds": {
            "validation_terminal_rgb_relative_change_max": -0.05,
            "local_test_terminal_rgb_relative_change_max": -0.03,
            "validation_reward_mae_ratio_max": 1.0,
            "local_test_reward_mae_ratio_max": 1.05,
            "validation_pairwise_win_rate_min": 0.75,
            "local_test_pairwise_win_rate_min": 0.65,
            "validation_normalized_margin_min": 0.05,
            "candidate_reward_change_min": 1e-6,
            "all_checks_required": True,
            "waivers_or_excluded_failed_checks_allowed": False,
        },
        "evidence": {
            str(path): sha256(path)
            for path in (script, runtime, parent, split, instruction_map, library, reward)
        },
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (REG / "preregistration.json").write_text(text)
    (RUN / "release_registration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
