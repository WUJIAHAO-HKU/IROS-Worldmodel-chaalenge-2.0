#!/usr/bin/env python3
"""Preregister v326 before contract or reward results exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v326_v317_blended_phase_terminal_gate_seed1496_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "parent_runtime": B / "pipeline/wam_pipeline/v324_phase_guarded_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_wrapper": B / "pipeline/scripts/test_v326_blended_phase_terminal.py",
        "inherited_contract": B / "pipeline/scripts/test_v324_phase_guarded_terminal.py",
        "audit_wrapper": B / "pipeline/scripts/audit_v326_blended_phase_terminal_gate.py",
        "inherited_audit": B / "pipeline/scripts/audit_v324_phase_guarded_terminal_gate.py",
        "metric_helper": B / "pipeline/scripts/audit_v325_same_episode_phase_repair_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "parent_rl_authorization": J / "v317_v315_native_batch_causal_gate_seed1490_20260822/audit/expensive_rl_authorization.json",
        "rejected_v324": J / "v324_v317_phase_guarded_terminal_gate_seed1494_20260822/audit/causal_gate_report.json",
        "rejected_v325": J / "v325_v317_same_episode_phase_repair_gate_seed1495_20260822/audit/causal_gate_report.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["parent_rl_authorization"].read_text())["launch_permission"] is not True:
        raise RuntimeError("v317 parent was not authorized")
    if json.loads(paths["phase_training_report"].read_text())["passed"] is not True:
        raise RuntimeError("v323 phase training did not pass")
    for name in ("rejected_v324", "rejected_v325"):
        if json.loads(paths[name].read_text())["passed"] is not False:
            raise RuntimeError(f"{name} was not rejected")
    payload = {
        "format": "strict-track2-v326-blended-phase-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v326-blended-phase-terminal-v317",
            "parent": "frozen authorized v317 plus v324 public phase/retrieval gates",
            "change": "reduce only v324 repair injections from alpha 1.0 to a fixed alpha 0.90; all gates and target selection remain frozen",
            "repair_alpha": 0.90,
            "deployment_batch_size": 8,
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "measurement_rule": {
            "byte_identical_split": "if every candidate terminal is byte-identical to v317, reuse v317 reward fidelity for that split",
            "changed_prediction_waiver": False,
        },
        "fixed_thresholds": {
            "transition_windows_min_each_split": 12,
            "validation_positive_reward_mean_min": 0.98,
            "local_test_positive_reward_mean_min": 0.90,
            "validation_positive_hit_rate_min": 0.90,
            "local_test_positive_hit_rate_min": 0.90,
            "validation_transition_reward_gain_min": -0.02,
            "local_test_transition_reward_gain_min": 0.20,
            "failure_hit_rate_max_validation": 0.10,
            "failure_hit_rate_max_local_test": 0.15,
            "pairwise_win_rate_min_validation": 0.90,
            "pairwise_win_rate_min_local_test": 0.85,
            "mean_margin_min_validation": 0.70,
            "mean_margin_min_local_test": 0.40,
            "all_window_rgb_mae_ratio_max_validation": 1.01,
            "all_window_rgb_mae_ratio_max_local_test": 1.03,
            "all_window_reward_mae_ratio_max_validation": 1.0,
            "all_window_reward_mae_ratio_max_local_test": 0.90,
            "left_bit_exact": True,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "public_episode_disjoint_data_only": True,
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
