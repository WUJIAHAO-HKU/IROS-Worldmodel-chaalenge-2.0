#!/usr/bin/env python3
"""Produce a requirement-by-requirement Track 2 completion verdict.

Missing evidence is pending, while present but contradictory evidence is
failed.  The goal can be complete only when every independently named gate is
passed; this script deliberately does not infer success from process intent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_VERSION = "track2-v15.7-hybrid-action-gated-reward-safe-blend12"
EXPERIMENT = "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"


def load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(name: str, status: str, evidence: Any, reason: str) -> dict[str, Any]:
    if status not in {"passed", "pending", "failed"}:
        raise ValueError(status)
    return {"name": name, "status": status, "reason": reason, "evidence": evidence}


def frozen_world_model(root: Path, audit: Path, joint: Path) -> dict[str, Any]:
    release = joint / "v157_hybrid_gate_blend12_formal_release"
    manifest_path = release / "release_manifest.json"
    adaptation_path = release / "onpolicy_adaptation/adaptation_manifest.json"
    verification_path = audit / "service_acceptance/v157_frozen_manifest_verification.json"
    manifest = load(manifest_path)
    adaptation = load(adaptation_path)
    verification = load(verification_path)
    if manifest is None or adaptation is None or verification is None:
        return gate(
            "frozen_v15_world_model",
            "pending",
            [str(manifest_path), str(adaptation_path), str(verification_path)],
            "Frozen release or verification evidence is missing.",
        )
    mismatches: list[str] = []
    for relative, expected in manifest.get("sha256", {}).items():
        path = release / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(str(path))
    for relative, expected in adaptation.get("sha256", {}).items():
        path = adaptation_path.parent / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(str(path))
    base_expected = adaptation.get("base_release_manifest_sha256")
    if base_expected != sha256(manifest_path):
        mismatches.append(str(manifest_path) + "#adaptation-base")
    verification_ok = (
        verification.get("checked_file_count") == len(verification.get("files", []))
        and verification.get("checked_file_count", 0) > 0
        and all(row.get("matches") is True for row in verification.get("files", []))
    )
    if mismatches or not verification_ok:
        return gate(
            "frozen_v15_world_model",
            "failed",
            {"mismatches": mismatches, "verification_ok": verification_ok},
            "At least one frozen release artifact differs from its manifest.",
        )
    return gate(
        "frozen_v15_world_model",
        "passed",
        {
            "release": str(release),
            "base_files": len(manifest["sha256"]),
            "adaptation_files": len(adaptation["sha256"]),
            "verification_files": verification["checked_file_count"],
        },
        "Base release, V15.7 adaptation and independent verification hashes match.",
    )


def official_flow_sources(root: Path, audit: Path) -> dict[str, Any]:
    compliance_path = root / "pipeline/config/strict_track2_runtime_source_compliance_audit.json"
    compliance = load(compliance_path)
    if compliance is None:
        return gate("official_unmodified_pi05_reward_grpo", "pending", str(compliance_path), "Source audit is missing.")
    runtime = audit / "full_budget_gpu_cache_release_runtime"
    mismatches: list[str] = []
    for relative, expected in compliance.get("exact_official_files", {}).items():
        path = runtime / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(str(path))
    for relative, row in compliance.get("declared_resource_only_files", {}).items():
        path = runtime / relative
        if not path.is_file() or sha256(path) != row.get("base_sha256"):
            mismatches.append(str(path))
    adapter = compliance.get("track2_interface_adapter", {})
    adapter_path = runtime / str(adapter.get("path", ""))
    if not adapter_path.is_file() or sha256(adapter_path) != adapter.get("base_sha256"):
        mismatches.append(str(adapter_path))
    pi_path = root / "artifacts/official_resources/pi05_adjust_bottle/model.safetensors"
    reward_path = root / "artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
    weight_hashes = compliance.get("weights", {})
    for path, key in ((pi_path, "pi05_model_sha256"), (reward_path, "reward_model_sha256")):
        if not path.is_file() or sha256(path) != weight_hashes.get(key):
            mismatches.append(str(path))
    participant_action_selection = adapter.get("participant_action_selection")
    if participant_action_selection is not False:
        mismatches.append("participant_action_selection")
    reward_equivalence_path = audit / "http_reward_equivalence_audit.json"
    reward_equivalence = load(reward_equivalence_path)
    if reward_equivalence is None or reward_equivalence.get("passed") is not True:
        mismatches.append(str(reward_equivalence_path))
    budget_provenance_path = audit / "public_budget_provenance_audit.json"
    budget_provenance = load(budget_provenance_path)
    if budget_provenance is None or budget_provenance.get("passed") is not True:
        mismatches.append(str(budget_provenance_path))
    status = "failed" if mismatches else "passed"
    return gate(
        "official_unmodified_pi05_reward_grpo",
        status,
        {
            "official_commit": compliance.get("official_commit"),
            "mismatches": mismatches,
            "pi05_sha256": weight_hashes.get("pi05_model_sha256"),
            "reward_sha256": weight_hashes.get("reward_model_sha256"),
            "participant_action_selection": participant_action_selection,
            "http_reward_equivalence": str(reward_equivalence_path),
            "http_reward_equivalence_passed": (
                reward_equivalence is not None and reward_equivalence.get("passed") is True
            ),
            "public_budget_provenance": str(budget_provenance_path),
            "budget_classification": (
                None
                if budget_provenance is None
                else budget_provenance.get("classification", {}).get("local_1000_epoch_run")
            ),
            "organizer_hidden_budget_claimed_equal": False,
        },
        "Official sources/weights and the no-participant-action-selection boundary match."
        if not mismatches
        else "Official source, weight or action-selection evidence contradicts the frozen audit.",
    )


def formal_policy(root: Path, audit: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = audit / f"runs/formal_full_budget_{MODEL_VERSION}_seed1244"
    checkpoint = run / EXPERIMENT / "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
    initialized = run / "audit/formal_run_initialized"
    contract = load(root / "pipeline/config/strict_track2_full_official_budget_contract.json")
    log = audit / "formal_v157_seed1244_automatic_launcher.log"
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    no_mpc_ready = initialized.is_file() and contract is not None
    no_mpc = gate(
        "no_participant_mpc",
        "passed" if no_mpc_ready else "pending",
        {
            "formal_initialized": initialized.is_file(),
            "prohibited": None if contract is None else contract.get("prohibited"),
        },
        "Formal seed1244 run was initialized under the contract that prohibits participant MPC."
        if no_mpc_ready
        else "The formal run has not yet initialized under the no-MPC contract.",
    )
    if not checkpoint.is_file() or "formal_full_budget_complete" not in log_text:
        policy = gate(
            "formal_1000_epoch_policy_checkpoint",
            "pending",
            {"checkpoint": str(checkpoint), "exists": checkpoint.is_file()},
            "A complete public full-reference-budget global_step_1000 checkpoint is not yet available.",
        )
    else:
        policy = gate(
            "formal_1000_epoch_policy_checkpoint",
            "passed",
            {"checkpoint": str(checkpoint), "bytes": checkpoint.stat().st_size},
            "The unoverridden public full-reference run reached global_step_1000 and exported full weights; this is not an organizer-certified held-out result.",
        )
    return no_mpc, policy


def real_robotwin(audit: Path) -> dict[str, Any]:
    report_path = audit / "real_robotwin_eval/formal_v157_seed1244_acceptance128_summary.json"
    report = load(report_path)
    variant = "formal_v157_seed1244_step1000"
    if report is None:
        return gate("real_robotwin_dual_arm_at_least_3_percent", "pending", str(report_path), "Sealed acceptance128 has not completed.")
    candidates = [row for row in report.get("candidates", []) if row.get("variant") == variant]
    if len(candidates) != 1:
        return gate("real_robotwin_dual_arm_at_least_3_percent", "failed", str(report_path), "Candidate metrics are missing or ambiguous.")
    comparison = candidates[0].get("comparison", {})
    success = comparison.get("success_delta", {})
    grasp = comparison.get("grasp_delta", {})
    checks = {
        "seed_count_128": report.get("seed_count") == 128,
        "selected": report.get("selected_variant") == variant,
        "eligible": comparison.get("eligible") is True,
        "absolute_gain_at_least_3pp": success.get("all", {}).get("absolute_at_least_3pp") is True,
        "relative_gain_at_least_3_percent": success.get("all", {}).get("relative_at_least_3_percent") is True,
        "left_success_non_regression": success.get("left", {}).get("absolute_percentage_points", -1) >= 0,
        "right_success_non_regression": success.get("right", {}).get("absolute_percentage_points", -1) >= 0,
        "all_grasp_non_regression": all(
            grasp.get(scope, {}).get("absolute_percentage_points", -1) >= 0
            for scope in ("all", "left", "right")
        ),
    }
    passed = all(checks.values())
    return gate(
        "real_robotwin_dual_arm_at_least_3_percent",
        "passed" if passed else "failed",
        {"report": str(report_path), "checks": checks, "success_delta": success, "grasp_delta": grasp},
        "Paired, disjoint acceptance128 proves the aggregate >=3% gate with both arms and grasp non-regressing."
        if passed
        else "The real RoboTwin acceptance report does not satisfy every preregistered policy gate.",
    )


def real_service(root: Path, audit: Path) -> dict[str, Any]:
    protocol_path = audit / "service_acceptance/real_v157_protocol_runtime_opt_v2_before_formal_seed1244.json"
    restart_path = audit / "service_acceptance/real_v157_restart_comparison_runtime_opt_v2_formal_seed1244.json"
    protocol, restart = load(protocol_path), load(restart_path)
    cache_path = audit / "service_acceptance/v157_retrieval_cache_equivalence.json"
    cache = load(cache_path)
    runtime_equivalence_path = audit / "service_acceptance/v157_deadwork_target_cache_equivalence.json"
    runtime_equivalence = load(runtime_equivalence_path)
    if protocol is None or restart is None or cache is None or runtime_equivalence is None:
        return gate(
            "real_v15_service_contract_restart_latency",
            "pending",
            [
                str(protocol_path),
                str(restart_path),
                str(cache_path),
                str(runtime_equivalence_path),
            ],
            "Current runtime's protocol, cache-equivalence or restart evidence is missing.",
        )
    tests = protocol.get("tests", {})
    required = {
        "authentication",
        "max_batch_8",
        "batch_9_rejected",
        "concurrency_1_overload_429",
        "idempotent_retry",
        "idempotency_conflict",
        "same_seed_same_pixels",
        "payload_over_32_mib",
    }
    checks = {
        "protocol_passed": protocol.get("passed") is True,
        "required_tests_passed": all(tests.get(name) == "passed" for name in required),
        "latency_under_600s": protocol.get("latency_ms", {}).get("all_under_recommended_timeout") is True,
        "restart_passed": restart.get("passed") is True,
        "restart_capabilities_identical": restart.get("capabilities_sha256_before") == restart.get("capabilities_sha256_after"),
        "restart_pixels_identical": restart.get("pixel_sha256_before") == restart.get("pixel_sha256_after"),
        "retrieval_cache_pixel_exact": cache.get("passed") is True
        and cache.get("checks", {}).get("pixels_exact") is True,
        "runtime_optimizations_pixel_exact": runtime_equivalence.get("passed") is True
        and runtime_equivalence.get("checks", {}).get("cold_pixels_exact") is True
        and runtime_equivalence.get("checks", {}).get("hot_pixels_exact") is True,
        "runtime_sources_still_match": bool(
            runtime_equivalence.get("runtime_source_sha256")
        )
        and all(
            (root / relative).is_file()
            and sha256(root / relative) == expected
            for relative, expected in runtime_equivalence.get(
                "runtime_source_sha256", {}
            ).items()
        ),
    }
    passed = all(checks.values())
    return gate(
        "real_v15_service_contract_restart_latency",
        "passed" if passed else "failed",
        {
            "protocol": str(protocol_path),
            "restart": str(restart_path),
            "retrieval_cache": str(cache_path),
            "runtime_optimization_equivalence": str(runtime_equivalence_path),
            "checks": checks,
        },
        "Actual V15.7 backend passes auth, max batch, concurrency, idempotency, restart and latency gates."
        if passed
        else "Actual V15.7 service evidence is present but one or more gates failed.",
    )


def public_https(audit: Path) -> dict[str, Any]:
    path = audit / "service_acceptance/real_v157_public_https_acceptance.json"
    report = load(path)
    if report is None:
        return gate("public_https_endpoint", "pending", str(path), "Public HTTPS end-to-end evidence is missing.")
    checks = {
        "passed": report.get("passed") is True,
        "scheme_https": report.get("scheme") == "https",
        "publicly_reachable": report.get("publicly_reachable") is True,
        "certificate_verified": report.get("certificate_verified") is True,
        "authentication_passed": report.get("authentication_passed") is True,
    }
    passed = all(checks.values())
    return gate(
        "public_https_endpoint",
        "passed" if passed else "failed",
        {"report": str(path), "checks": checks},
        "A publicly reachable, certificate-verified HTTPS endpoint passes authenticated requests."
        if passed
        else "HTTPS evidence exists but does not prove every public endpoint requirement.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = root / "artifacts/strict_track2_official_20260810"
    joint = root / "artifacts/strict_track2_joint_augmentation_20260810"
    no_mpc, policy = formal_policy(root, audit)
    gates = [
        frozen_world_model(root, audit, joint),
        official_flow_sources(root, audit),
        no_mpc,
        policy,
        real_robotwin(audit),
        real_service(root, audit),
        public_https(audit),
    ]
    report = {
        "format": "strict-track2-goal-completion-audit-v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "model_version": MODEL_VERSION,
        "goal_complete": all(row["status"] == "passed" for row in gates),
        "counts": {
            status: sum(row["status"] == status for row in gates)
            for status in ("passed", "pending", "failed")
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
