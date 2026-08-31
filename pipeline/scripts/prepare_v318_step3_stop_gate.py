#!/usr/bin/env python3
"""Preregister the training-only global-step-3 continuation gate for v318."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = OFF / "runs" / NAME
    reg = OFF / "run_registry" / NAME
    output = reg / "global_step3_continuation_gate_preregistration_v3.json"
    if output.exists():
        raise FileExistsError(output)

    inputs = {
        "training_preregistration": reg / "preregistration.json",
        "world_model_authorization": (
            ROOT
            / "artifacts/strict_track2_joint_augmentation_20260810"
            / "v317_v315_native_batch_causal_gate_seed1490_20260822"
            / "audit/expensive_rl_authorization.json"
        ),
        "offline_policy_auditor": ROOT / "pipeline/scripts/audit_v318_offline_policy_trainfit.py",
        "continuation_auditor": ROOT / "pipeline/scripts/audit_v318_step3_stop_gate.py",
    }
    for path in inputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    payload = {
        "format": "strict-track2-v318-global-step3-continuation-gate-v3",
        "supersedes": {
            "path": str(reg / "global_step3_continuation_gate_preregistration_v2.json"),
            "reason": "v2 referenced the historical auditor whose API no longer matches the available legacy PT loader; v3 pins a one-line num_workers compatibility copy before any offline diagnostic result was produced",
        },
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "decision_point": {
            "checkpoint": "global_step_3",
            "completed_metric_steps": [0, 1, 2],
            "action_if_passed": "resume the preregistered fixed-budget run from global_step_3",
            "action_if_failed": "reject v318 and do not spend steps 3 through 9",
        },
        "thresholds": {
            "step2_success_once_min": 0.03125,
            "first3_success_once_sum_min": 0.15625,
            "first3_return_mean_min": 0.075,
            "left_route_accuracy_min": 1.0,
            "right_route_accuracy_min": 1.0,
            "candidate_sample_mse_over_official_max": 1.5,
            "candidate_right_active_mse_over_official_max": 1.5,
            "candidate_inactive_mse_over_official_max": 2.0,
        },
        "rationale": {
            "step2_success_once_min": "requires recovery to at least the observed step1 coverage of 4/128",
            "first3_success_once_sum_min": "requires at least 20 success-once trajectories over 384 rollouts",
            "first3_return_mean_min": "rejects reward-strength collapse while allowing fixed-seed sampling variance",
            "offline_policy_gate": "public train40 only; rejects arm-routing collapse or gross action-distribution drift",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()
        },
        "guards": {
            "public_train40_actions_only_for_offline_diagnostic": True,
            "public112_outcomes_read": False,
            "reserved_final128_outcomes_read": False,
            "hidden_outcomes_read": False,
            "real_submission": False,
            "algorithm_or_budget_changed": False,
            "thresholds_registered_before_step2_metrics_or_checkpoint_diagnostic": True,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
