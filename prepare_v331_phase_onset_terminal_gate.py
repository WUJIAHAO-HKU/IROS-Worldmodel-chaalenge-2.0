#!/usr/bin/env python3
"""Preregister v331 visible onset semantics and all gates before evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v331_v326_phase_onset_terminal_gate_seed1501_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v331_phase_onset_terminal_runtime.py",
        "v326_parent_runtime": B / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract": B / "pipeline/scripts/test_v331_phase_onset_terminal.py",
        "inherited_contract": B / "pipeline/scripts/test_v324_phase_guarded_terminal.py",
        "causal_audit": B / "pipeline/scripts/audit_v331_phase_onset_terminal_gate.py",
        "inherited_causal_audit": B / "pipeline/scripts/audit_v324_phase_guarded_terminal_gate.py",
        "recursive_audit": B / "pipeline/scripts/audit_v331_public_recursive_stability.py",
        "inherited_recursive_audit": B / "pipeline/scripts/audit_v328_public_recursive_stability.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "v328_failed_contract": J / "v328_v326_coherent_phase_trajectory_gate_seed1498_20260822/audit/contract_report.json",
        "v330_failed_sweep": J / "v330_train_phase_proxy_terminal_bridge_seed1500_20260822/audit/sweep_report.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["v326_authorization"].read_text()).get("launch_permission") is not True:
        raise RuntimeError("v326 parent was not authorized")
    if json.loads(paths["v328_failed_contract"].read_text()).get("passed") is not False:
        raise RuntimeError("v328 rejection is not frozen")
    if json.loads(paths["v330_failed_sweep"].read_text()).get("selection_succeeded") is not False:
        raise RuntimeError("v330 must fail closed before v331")
    phase = json.loads(paths["phase_training_report"].read_text())
    if phase.get("passed") is not True:
        raise RuntimeError("phase table training gate failed")
    payload = {
        "format": "strict-track2-v331-phase-onset-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v331-phase-onset-terminal-v326-v317",
            "parent": "frozen v326/v317 action, phase, alpha, splice, and failure semantics",
            "change": (
                "on an already-authorized v326 repair only, select the first frozen sustained-success "
                "onset row; prefer the matched base episode when its terminal is eligible, otherwise "
                "select an eligible onset by the frozen endpoint/visual metric"
            ),
            "unchanged": "repair alpha 0.90, first seven v317 frames, all gates, counterfactual suppression",
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "hypothesis": {
            "v326_problem": "eligible terminal tails are mostly blank after the bottle and arm leave view",
            "onset_property": "v323 onset is the first of at least three consecutive public-train rows scoring >=0.90",
            "intended_effect": "retain a visible grasped bottle/arm while preserving official reward and scene identity",
        },
        "fixed_thresholds": {
            "contract_all_checks_required": True,
            "transition_next_context_rgb_strictly_improves_v326": True,
            "transition_temporal_delta_strictly_improves_v326": True,
            "transition_foreground_fraction_gain_min": 0.05,
            "counterfactual_full_sequence_v317_bit_exact": True,
            "inherited_v326_terminal_causal_checks_all_required": True,
            "local_all_window_rgb_ratio_max": 1.03,
            "local_all_window_reward_error_ratio_max": 1.0,
            "recursive_phase_qualified_windows_min_each_split": 2,
            "recursive_phase_rgb_mae_ratio_max_vs_v326_each_split": 0.90,
            "recursive_phase_temporal_delta_error_ratio_max_vs_v326_each_split": 0.90,
            "recursive_phase_reward_error_ratio_max_vs_v326_each_split": 1.0,
            "recursive_all_rgb_mae_ratio_max_vs_v326_each_split": 1.02,
            "recursive_all_reward_error_ratio_max_vs_v326_each_split": 1.02,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "evaluation_sequence": [
            "numeric_visibility_and_nonregression_contract",
            "frozen_terminal_causal_reward_gate",
            "public_heldout_recursive_stability_gate",
            "expensive_RL_authorization_only_if_every_gate_passes",
        ],
        "evidence": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "declared_public_world_model_data_only": True,
            "public_evaluation_outcomes": False,
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
