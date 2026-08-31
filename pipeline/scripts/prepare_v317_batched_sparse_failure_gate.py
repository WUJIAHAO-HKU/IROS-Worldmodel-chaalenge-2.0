#!/usr/bin/env python3
"""Preregister native-batched v317 before contract or reward results exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v317_v315_native_batch_causal_gate_seed1490_20260822"
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
        "runtime": B / "pipeline/wam_pipeline/v317_batched_sparse_failure_terminal_runtime.py",
        "serial_parent_runtime": B / "pipeline/wam_pipeline/v315_sparse_failure_terminal_runtime.py",
        "backends": B / "pipeline/wam_pipeline/backends.py",
        "contract_test": B / "pipeline/scripts/test_v317_batched_sparse_failure_terminal.py",
        "audit": B / "pipeline/scripts/audit_v317_batched_sparse_failure_terminal_gate.py",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "training_report": J / "v311_public_action_causal_gate_seed1484_20260822/audit/training_report.json",
        "serial_causal_report": J / "v316_v315_hash_corrected_sparse_failure_gate_seed1489_20260822/audit/causal_gate_report.json",
        "serial_rl_authorization": J / "v316_v315_hash_corrected_sparse_failure_gate_seed1489_20260822/audit/expensive_rl_authorization.json",
        "split": B / "artifacts/splits/adjust_bottle_50episodes_full.json",
        "instruction_map": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not json.loads(paths["serial_rl_authorization"].read_text())["launch_permission"]:
        raise RuntimeError("serial parent was not authorized")
    payload = {
        "format": "strict-track2-v317-native-batch-causal-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_version": "track2-v317-batched-sparse-failure-terminal-v315",
            "single_request_semantics": "unchanged v315",
            "batch_semantics": "native v271 direct/mirror inference followed by frozen v315 sparse terminal rule",
            "deployment_batch_size": 8,
            "failure_probability_threshold": 0.01,
            "runtime_reward_or_outcome_access": False,
            "response": "eight RGB frames only",
        },
        "numeric_contract": {
            "comparison": "v317 native batch versus v315 serial requests with TF32 disabled",
            "max_absolute_pixel_change_max": 2,
            "mean_absolute_pixel_change_max": 0.05,
            "native_batch_speedup_min": 1.5,
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
            "positive_terminal_parent_max_pixel_difference": 2,
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
