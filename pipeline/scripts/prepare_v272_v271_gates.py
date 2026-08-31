#!/usr/bin/env python3
"""Preregister public-only service and long-horizon gates for v271."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = B / "artifacts/strict_track2_official_20260810"
J = B / "artifacts/strict_track2_joint_augmentation_20260810"
N = "v272_v271_endpoint_calibrated_gates_seed1469_20260819"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = O / "run_registry" / N
    run = J / N
    if registry.exists() or run.exists():
        raise FileExistsError("refusing overwrite")
    policy_fit = O / "diagnostics/v269_offline_policy_trainfit_20260819.json"
    endpoint = O / "diagnostics/v270_endpoint_progress_gate_20260819_v3.json"
    fit = json.loads(policy_fit.read_text())
    endpoint_report = json.loads(endpoint.read_text())
    if fit["rules"]["evaluation_batches_accessed"] != []:
        raise RuntimeError("policy-fit audit was not training-data-only")
    if endpoint_report["guards"]["hidden_or_final_data"]:
        raise RuntimeError("endpoint audit used protected data")
    runtime = B / "pipeline/wam_pipeline/v271_endpoint_calibrated_terminal_runtime.py"
    sources = [
        runtime,
        B / "pipeline/wam_pipeline/v254_delta_regime_terminal_runtime.py",
        B / "pipeline/wam_pipeline/backends.py",
        B / "pipeline/tests/test_v271_endpoint_calibrated_terminal_runtime.py",
        B / "pipeline/scripts/replay_v245_post_grasp_reward.py",
        B / "pipeline/scripts/restart_v271_services.sh",
        B / "pipeline/scripts/launch_v272_v271_gates.sh",
    ]
    registry.mkdir(parents=True)
    (run / "audit").mkdir(parents=True)
    (run / "local_dev_token.txt").write_text("local-dev-token\n")
    capture_gate = run / "audit/post_grasp_reward_report.json"
    payload = {
        "format": "strict-track2-v272-v271-gates-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "fixed_model": {
            "model_version": "track2-v271-endpoint-calibrated-terminal",
            "inherits_v254_clean_and_mid_regime": True,
            "endpoint_temperature": 0.25,
            "relative_progress_floor": -0.25,
            "relative_progress_ceiling": 0.50,
            "ood_alpha_boost": 2.0,
            "target_rule": "public-clean terminal successor",
        },
        "fixed_gate": {
            "success_like_min": 4,
            "group_std_min": 0.005,
            "alignment_global_min": 0.05,
            "alignment_group_min": 0.40,
            "service_contract": True,
        },
        "input_capture_gate": str(capture_gate),
        "fixed_audit": {
            "chunks": 16,
            "threshold": 0.9,
            "success_hit_rate_min": 0.75,
            "failure_hit_rate_max": 0.20,
            "margin_min": 0.55,
            "official_http_acceptance": True,
        },
        "evidence": {
            "policy_fit": str(policy_fit),
            "policy_fit_sha256": sha(policy_fit),
            "endpoint_analysis": str(endpoint),
            "endpoint_analysis_sha256": sha(endpoint),
            "v169_to_v265_right_flow_loss_reduction": 1.0 - fit["checkpoints"]["v265"]["metrics"]["right"]["flow_loss_first8_physical14"] / fit["checkpoints"]["v169"]["metrics"]["right"]["flow_loss_first8_physical14"],
            "v254_ood_reward_vs_negative_endpoint_distance": -0.15397431796154543,
            "calibrated_nonzero_alpha_endpoint_correlation": 0.560718840020133,
            "calibrated_nonzero_alpha_progress_correlation": 0.6074630177397305,
        },
        "implementation": {str(path): sha(path) for path in sources},
        "guards": {
            "runtime_uses_reward": False,
            "public_data_only": True,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (registry / "preregistration.json").write_text(text)
    (run / "release_registration.json").write_text(text)


if __name__ == "__main__":
    main()

