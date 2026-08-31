#!/usr/bin/env python3
"""Preregister v312 implementation and held-out hard gates before evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v312_v311_causal_terminal_runtime_gate_seed1485_20260822"
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
        raise FileExistsError("refusing overwrite")
    paths = {
        "runtime": B / "pipeline/wam_pipeline/v312_causal_terminal_mirror_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_test": B / "pipeline/scripts/test_v312_causal_terminal_mirror.py",
        "audit": B / "pipeline/scripts/audit_v312_causal_runtime_gate.py",
        "v310_helpers": B / "pipeline/scripts/audit_v310_full_mirror_causal_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_gate_training": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    training = json.loads(paths["action_gate_training"].read_text())
    if training.get("passed") is not True:
        raise RuntimeError("v311 training gate did not pass")
    payload = {
        "format": "strict-track2-v312-causal-terminal-runtime-gate-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v312-public-action-gated-full-terminal-mirror-v271",
            "right_expert_like": "full mirrored v271 eight-frame successor",
            "right_rejected": "mirrored parametric parent only, no successful retrieval target",
            "left": "bit-exact direct v271",
            "response": "eight RGB frames only",
        },
        "fixed_data": {
            "decision_episodes": [7, 18],
            "confirmation_episodes": [6, 22],
            "samples_per_split": 64,
            "counterfactuals": ["open_gripper", "static_transport", "reverse_transport"],
            "episode_disjoint": True,
            "hidden_or_final_data": False,
        },
        "fixed_thresholds": {
            "validation_terminal_rgb_relative_change_max": -0.05,
            "local_test_terminal_rgb_relative_change_max": -0.03,
            "validation_reward_mae_ratio_max": 1.0,
            "local_test_reward_mae_ratio_max": 1.05,
            "validation_pairwise_win_rate_min": 0.75,
            "local_test_pairwise_win_rate_min": 0.65,
            "validation_normalized_margin_min": 0.05,
            "validation_positive_accept_rate_min": 0.70,
            "local_test_positive_accept_rate_min": 0.60,
            "validation_negative_reject_rate_min": 0.80,
            "local_test_negative_reject_rate_min": 0.70,
            "left_bit_exact": True,
            "serial_batch_bit_exact": True,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "runtime_reads_reward_or_outcome": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
