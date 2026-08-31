#!/usr/bin/env python3
"""Preregister v322 before contract or reward results exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v322_v317_causal_alpha_terminal_gate_seed1492_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v322_causal_alpha_terminal_runtime.py",
        "parent_runtime": B / "pipeline/wam_pipeline/v317_batched_sparse_failure_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_test": B / "pipeline/scripts/test_v322_causal_alpha_terminal.py",
        "audit": B / "pipeline/scripts/audit_v322_causal_alpha_terminal_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "parent_causal_report": J / "v317_v315_native_batch_causal_gate_seed1490_20260822/audit/causal_gate_report.json",
        "parent_rl_authorization": J / "v317_v315_native_batch_causal_gate_seed1490_20260822/audit/expensive_rl_authorization.json",
        "rejected_lookup_diagnostic": J / "v321_action_episode_terminal_diagnostic_seed1491_20260822/audit/diagnostic_report.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not json.loads(paths["parent_rl_authorization"].read_text())["launch_permission"]:
        raise RuntimeError("v317 parent was not authorized")
    if json.loads(paths["rejected_lookup_diagnostic"].read_text())["passed"] is not False:
        raise RuntimeError("v321 diagnostic was not rejected")
    payload = {
        "format": "strict-track2-v322-causal-alpha-terminal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v322-causal-alpha-terminal-v317",
            "parent": "frozen authorized v317",
            "change": "for valid post-grasp right requests only, select a public terminal episode by normalized endpoint distance plus 0.25 visual distance and raise v271 terminal alpha with the frozen v311 action-gate probability mapped from 0.90..0.99 to 0..1",
            "deployment_batch_size": 8,
            "failure_probability_threshold": 0.01,
            "terminal_episode_visual_weight": 0.25,
            "causal_alpha_probability_floor": 0.90,
            "causal_alpha_probability_ceil": 0.99,
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "numeric_contract": {
            "comparison": "v322 native batch versus v322 serial requests with TF32 disabled",
            "max_absolute_pixel_change_max": 2,
            "mean_absolute_pixel_change_max": 0.05,
            "native_batch_speedup_min": 1.4,
            "left_single_request_parent_bit_exact": True,
            "counterfactual_terminal_context_bit_exact": True,
        },
        "causal_audit": {
            "source": "all windows from public validation episodes 7/18 and confirmation episodes 6/22",
            "inference_path": "deployed native predict_batch with batch size 8",
            "transition_definition": "frozen official reward(context)<=0.10 and reward(public GT terminal)>=0.90",
            "counterfactuals": ["open_gripper", "static_transport", "reverse_transport"],
        },
        "fixed_thresholds": {
            "transition_windows_min_each_split": 12,
            "validation_positive_reward_mean_min": 0.95,
            "local_test_positive_reward_mean_min": 0.85,
            "validation_positive_hit_rate_min": 0.90,
            "local_test_positive_hit_rate_min": 0.80,
            "validation_transition_reward_gain_min": -0.02,
            "local_test_transition_reward_gain_min": 0.20,
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
            "all_window_rgb_mae_ratio_max_each_split": 1.05,
            "all_window_reward_mae_ratio_validation_max": 1.0,
            "all_window_reward_mae_ratio_local_test_max": 1.0,
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
