#!/usr/bin/env python3
"""Preregister v335 and the corrected all-offset recursive gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v335_v334_all_offset_recursive_gate_seed1505_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v335_all_offset_bounded_onset_runtime.py",
        "v334_parent_runtime": B / "pipeline/wam_pipeline/v334_bounded_onset_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract": B / "pipeline/scripts/test_v335_all_offset_bounded_onset.py",
        "causal_audit": B / "pipeline/scripts/audit_v335_bounded_onset_terminal_gate.py",
        "recursive_audit": B / "pipeline/scripts/audit_v335_all_offset_recursive_stability.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "v334_contract": J / "v334_v333_bounded_onset_terminal_gate_seed1504_20260822/audit/contract_report.json",
        "v334_causal": J / "v334_v333_bounded_onset_terminal_gate_seed1504_20260822/audit/causal_gate_report.json",
        "v334_recursive_failure_log": J / "v334_v333_bounded_onset_terminal_gate_seed1504_20260822/audit/recursive_stability.log",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["v334_contract"].read_text()).get("passed") is not True:
        raise RuntimeError("v334 contract did not pass")
    if json.loads(paths["v334_causal"].read_text()).get("passed") is not True:
        raise RuntimeError("v334 causal gate did not pass")
    if "empty aggregate" not in paths["v334_recursive_failure_log"].read_text():
        raise RuntimeError("v334 recursive audit bug is not frozen")
    payload = {
        "format": "strict-track2-v335-all-offset-bounded-onset-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v335-all-offset-bounded-onset-v334-v333-v326-v325-v317",
            "semantics": "bit-identical subclass of v334; only the recursive audit design changes",
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "audit_correction": {
            "old_bug": "stride-eight replay had zero validation changed rows and raised on an empty aggregate",
            "new_coverage": "all eight start modulo eight chains cover every one of 512 right holdout windows exactly once",
            "validation_rule": "candidate must be teacher-forced and recursively bit-exact to v326 because it is intentionally unchanged",
            "local_rule": "strict improvement is required only on structurally changed bounded-onset rows",
        },
        "fixed_thresholds": {
            "row_count": 512,
            "validation_changed_branch_count": 0,
            "local_changed_branch_count_min": 3,
            "validation_teacher_and_recursive_v326_bit_exact": True,
            "local_changed_recursive_rgb_ratio_max": 0.90,
            "local_changed_recursive_temporal_ratio_max": 0.98,
            "local_changed_recursive_reward_error_ratio_max": 1.50,
            "local_changed_recursive_reward_mean_min": 0.90,
            "local_changed_recursive_reward_hit_rate_min": 0.85,
            "local_all_recursive_rgb_ratio_max": 1.02,
            "local_all_recursive_temporal_ratio_max": 1.02,
            "local_all_recursive_reward_error_ratio_max": 1.02,
            "all_checks_required": True,
            "waivers_allowed": False,
        },
        "evaluation_sequence": [
            "repeat_v334_contract_under_v335_hash",
            "repeat_v334_causal_gate_under_v335_hash",
            "corrected_all_offset_public_recursive_gate",
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
