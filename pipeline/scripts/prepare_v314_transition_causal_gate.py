#!/usr/bin/env python3
"""Preregister v314 and its transition-specific causal thresholds."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v314_v311_transition_causal_gate_seed1487_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v314_transition_causal_terminal_runtime.py",
        "v312_base_runtime": B / "pipeline/wam_pipeline/v312_causal_terminal_mirror_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_test": B / "pipeline/scripts/test_v314_transition_causal_terminal.py",
        "base_contract_test": B / "pipeline/scripts/test_v312_causal_terminal_mirror.py",
        "audit": B / "pipeline/scripts/audit_v314_transition_causal_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "format": "strict-track2-v314-transition-causal-gate-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v314-transition-causal-terminal-v271",
            "right_gate_accepted": "mirrored frames 1--7 and direct v271 frame 8",
            "right_gate_rejected": "mirrored parametric frames 1--7 and observed context frame 8",
            "left": "bit-exact direct v271",
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "causal_audit": {
            "source": "declared public successful demonstrations",
            "decision_episodes": [7, 18],
            "confirmation_episodes": [6, 22],
            "samples_per_split": 64,
            "transition_definition": "frozen official reward(context)<=0.10 and reward(public GT terminal)>=0.90",
            "transition_definition_use": "offline audit only; never available to runtime",
            "counterfactuals": ["open_gripper", "static_transport", "reverse_transport"],
        },
        "fixed_thresholds": {
            "transition_windows_min_each_split": 12,
            "validation_positive_hit_rate_min": 0.75,
            "local_test_positive_hit_rate_min": 0.70,
            "validation_failure_hit_rate_max": 0.10,
            "local_test_failure_hit_rate_max": 0.15,
            "validation_pairwise_win_rate_min": 0.90,
            "local_test_pairwise_win_rate_min": 0.85,
            "validation_mean_margin_min": 0.70,
            "local_test_mean_margin_min": 0.60,
            "validation_positive_gate_accept_min": 0.90,
            "local_test_positive_gate_accept_min": 0.85,
            "validation_counterfactual_gate_reject_min": 0.90,
            "local_test_counterfactual_gate_reject_min": 0.85,
            "all_window_rgb_mae_ratio_max_each_split": 1.05,
            "all_window_reward_mae_ratio_validation_max": 1.0,
            "all_window_reward_mae_ratio_local_test_max": 1.05,
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
