#!/usr/bin/env python3
"""Preregister v334 bounded onset semantics and all gates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v334_v333_bounded_onset_terminal_gate_seed1504_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v334_bounded_onset_terminal_runtime.py",
        "v333_parent_runtime": B / "pipeline/wam_pipeline/v333_hybrid_onset_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract": B / "pipeline/scripts/test_v334_bounded_onset_terminal.py",
        "causal_audit": B / "pipeline/scripts/audit_v334_bounded_onset_terminal_gate.py",
        "recursive_audit": B / "pipeline/scripts/audit_v334_public_recursive_stability.py",
        "inherited_recursive_audit": B / "pipeline/scripts/audit_v328_public_recursive_stability.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "v333_failed_causal": J / "v333_v326_v325_hybrid_onset_terminal_gate_seed1503_20260822/audit/causal_gate_report.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["v326_authorization"].read_text()).get("launch_permission") is not True:
        raise RuntimeError("v326 parent was not authorized")
    failed = json.loads(paths["v333_failed_causal"].read_text())
    if failed.get("passed") is not False or failed["transition_summaries"]["local_test"]["positive_hit_rate_at_0p9"] < 0.90:
        raise RuntimeError("v333 near-pass evidence is invalid")
    payload = {
        "format": "strict-track2-v334-bounded-onset-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v334-bounded-onset-terminal-v333-v326-v325-v317",
            "parent": "frozen v333 hybrid on frozen v326/v325/v317",
            "change": (
                "same-episode onset is allowed only when the matched terminal-ineligible public "
                "row has phase_start - phase_onset in [0,2]; all later offsets use exact v326 fallback"
            ),
            "bound": 2,
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "frozen_diagnosis": {
            "desired_episode22_offsets": [0, 1],
            "undesired_episode6_late_offsets": [23, 24, 26, 27],
            "v333_reward_gate": "14/15 hits and mean 0.9072867 already passed",
            "v333_only_failed_gate": "local all-window RGB ratio 1.050995 > 1.03",
        },
        "fixed_thresholds": {
            "contract_all_checks_required": True,
            "late_phase_full_sequence_v326_bit_exact": True,
            "hybrid_next_context_rgb_strictly_improves_v326": True,
            "hybrid_temporal_delta_strictly_improves_v326": True,
            "inherited_v326_terminal_causal_checks_all_required": True,
            "local_all_window_rgb_ratio_max": 1.03,
            "local_all_window_reward_error_ratio_max": 0.90,
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
            "numeric_bounded_and_nonregression_contract",
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
