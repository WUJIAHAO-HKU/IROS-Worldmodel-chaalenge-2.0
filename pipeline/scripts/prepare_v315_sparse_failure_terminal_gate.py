#!/usr/bin/env python3
"""Preregister v315 sparse failure correction and all fail-closed gates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v315_v311_sparse_failure_terminal_gate_seed1488_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v315_sparse_failure_terminal_runtime.py",
        "v312_base_runtime": B / "pipeline/wam_pipeline/v312_causal_terminal_mirror_runtime.py",
        "v295_parent_runtime": B / "pipeline/wam_pipeline/v295_terminal_frame_preserving_mirror_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_test": B / "pipeline/scripts/test_v315_sparse_failure_terminal.py",
        "audit": B / "pipeline/scripts/audit_v315_sparse_failure_terminal_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "v315_exploration": J / "v314_v311_transition_causal_gate_seed1487_20260822/audit/v315_failure_signature_exploration.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "format": "strict-track2-v315-sparse-failure-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v315-sparse-failure-terminal-v271",
            "ordinary_requests": "bit-equivalent v295 branch semantics",
            "explicit_failure_terminal": "observed context RGB terminal",
            "failure_signatures": {
                "release_after_closed": "history right gripper <=0.5 and future mean >0.5",
                "static_transport": "right joint path <=1e-6",
                "reverse_like": "future right gripper mean <=0.5, moving, frozen v311 probability <0.01",
            },
            "left": "bit-exact direct v271",
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "threshold_selection": {
            "source": "v314 validation episodes 7 and 18 only",
            "chosen_probability_max": 0.01,
            "reason": "smallest tested decade detecting all three transition counterfactual classes; no additional valid-demo failure signatures versus lower thresholds",
            "local_test_use": "confirmation only; did not influence the fixed rule",
        },
        "causal_audit": {
            "source": "all windows from declared public successful demonstrations",
            "decision_episodes": [7, 18],
            "confirmation_episodes": [6, 22],
            "sampling": "all available windows; no outcome-based runtime routing",
            "transition_definition": "frozen official reward(context)<=0.10 and reward(public GT terminal)>=0.90",
            "transition_definition_use": "offline audit only; never available to runtime",
            "counterfactuals": ["open_gripper", "static_transport", "reverse_transport"],
        },
        "fixed_thresholds": {
            "transition_windows_min_each_split": 12,
            "positive_terminal_parent_bit_exact": True,
            "validation_failure_hit_rate_max": 0.10,
            "local_test_failure_hit_rate_max": 0.15,
            "validation_pairwise_win_rate_min": 0.90,
            "local_test_pairwise_win_rate_min": 0.85,
            "validation_mean_margin_min": 0.70,
            "local_test_mean_margin_min": 0.40,
            "validation_positive_route_accept_min": 0.90,
            "local_test_positive_route_accept_min": 0.85,
            "validation_counterfactual_suppression_min": 0.90,
            "local_test_counterfactual_suppression_min": 0.85,
            "all_window_rgb_mae_ratio_max_each_split": 1.0,
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
