#!/usr/bin/env python3
"""Preregister v328 code, data boundary, and all gates before measuring it."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v328_v326_coherent_phase_trajectory_gate_seed1498_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v328_coherent_phase_trajectory_runtime.py",
        "v326_parent_runtime": B / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "v317_parent_runtime": B / "pipeline/wam_pipeline/v317_batched_sparse_failure_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract": B / "pipeline/scripts/test_v328_coherent_phase_trajectory.py",
        "inherited_contract": B / "pipeline/scripts/test_v324_phase_guarded_terminal.py",
        "causal_audit": B / "pipeline/scripts/audit_v328_coherent_phase_trajectory_gate.py",
        "inherited_causal_audit": B / "pipeline/scripts/audit_v324_phase_guarded_terminal_gate.py",
        "recursive_audit": B / "pipeline/scripts/audit_v328_public_recursive_stability.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "action_training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "phase_gate": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "v326_authorization": J / "v326_v317_blended_phase_terminal_gate_seed1496_20260822/audit/expensive_rl_authorization.json",
        "v327_rejection": O / "runs/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/audit/training_go_no_go.json",
        "v327_recursive_diagnostic": O / "runs/v327_v169step5_v326_authorized_oneupdate_h200_r4_lr1e5_seed1497_20260822/audit/recursive_video_quality.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(paths["v326_authorization"].read_text()).get("launch_permission") is not True:
        raise RuntimeError("v326 parent was not authorized")
    rejection = json.loads(paths["v327_rejection"].read_text())
    if rejection.get("decision") != "reject_before_public_evaluation":
        raise RuntimeError("v327 training-only rejection is not frozen")
    diagnostic = json.loads(paths["v327_recursive_diagnostic"].read_text())
    if diagnostic.get("trajectory_count") != 128 or diagnostic["guards"].get("hidden_or_final_data_read") is not False:
        raise RuntimeError("v327 recursive diagnostic boundary is invalid")

    payload = {
        "format": "strict-track2-v328-coherent-phase-trajectory-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v328-coherent-phase-trajectory-v326-v317",
            "parent": "frozen authorized v326 terminal semantics on frozen v317",
            "change": (
                "retain all eight v326 direct frames only when the frozen v326 action/phase "
                "override is true and the frozen v315 failure signature is absent; otherwise "
                "retain v317 splice/suppression behavior bit-exactly"
            ),
            "closed_loop_context": "RLinf feeds prediction frames 3..7 into the next request",
            "repair_alpha": 0.90,
            "deployment_batch_size": 8,
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "hypothesis": {
            "observation": (
                "v327 rollout video changes from 68.6% near-white pixels initially to 16.9% "
                "at chunk25 while saturation grows 3.30x and gradient energy 4.73x"
            ),
            "mechanism": (
                "v317 batch semantics splice four mirrored frames and one direct v326 terminal "
                "into the exact five-frame context recursively consumed by the bridge"
            ),
            "intended_effect": (
                "remove that 4+1 context splice on the already-authorized phase branch without "
                "changing its terminal or any non-authorized request"
            ),
        },
        "fixed_thresholds": {
            "inherited_v326_terminal_checks_all_required": True,
            "serial_batch_equivalence": True,
            "left_bit_exact": True,
            "counterfactual_full_sequence_v317_bit_exact": True,
            "valid_next_context_rgb_strictly_improves_v317": True,
            "valid_next_context_temporal_delta_strictly_improves_v317": True,
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
            "numeric_and_coherence_contract",
            "frozen_terminal_causal_gate",
            "public_heldout_recursive_stability_gate",
            "expensive_RL_authorization_only_if_every_gate_passes",
        ],
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
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
