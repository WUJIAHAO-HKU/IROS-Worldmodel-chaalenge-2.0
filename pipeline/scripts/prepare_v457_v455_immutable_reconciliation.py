#!/usr/bin/env python3
"""Preregister a read-only reconciliation of the frozen v455 audit."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

SEED = 1610
EXPECTED_CHECKS = (
    "format", "generation_pass", "exact_fixed_four",
    "transport_each_three_contexts", "single_chunk_endpoint_source",
    "preprocess_closure", "runtime_provenance", "resource_contract",
    "endpoint_authority_only", "no_reward_policy",
    "selection_manifest_exact", "paired_endpoint_residual_only",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("v455-registry", "generator", "auditor", "resize-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    reg = args.v455_registry.resolve()
    prereg_path = reg / "preregistration.json"
    report_path = reg / "dataset" / "generation_report.json"
    legacy_audit_path = reg / "audit.json"
    npz_paths = sorted((reg / "dataset").glob("episode*_start*.npz"))
    if len(npz_paths) != 4:
        raise RuntimeError(f"v457 requires exactly four frozen v455 NPZs, got {len(npz_paths)}")
    prereg = json.loads(prereg_path.read_text())
    report = json.loads(report_path.read_text())
    legacy = json.loads(legacy_audit_path.read_text())
    if prereg.get("format") != "strict-track2-v455-postclose-endpoint-pilot-preregistration-v1":
        raise RuntimeError("unexpected frozen v455 preregistration")
    if report.get("format") != "strict-track2-v455-postclose-endpoint-pilot-generation-report-v1" or report.get("passed") is not True:
        raise RuntimeError("frozen v455 generation did not pass")
    checks = legacy.get("checks", {})
    if tuple(checks) != EXPECTED_CHECKS or sum(value is True for value in checks.values()) != 11:
        raise RuntimeError(f"unexpected v455 audit check profile: {checks}")
    if {name for name, value in checks.items() if value is False} != {"paired_endpoint_residual_only"}:
        raise RuntimeError("v455 must have exactly the known legacy-key failure")
    report_guards = report.get("guards", {})
    if report_guards.get("factual_public_endpoint_mae_diagnostic_only") is not True:
        raise RuntimeError("canonical endpoint-MAE diagnostic guard is absent")
    if "factual_public_fidelity_diagnostic_only" in report_guards:
        raise RuntimeError("legacy comparator key unexpectedly exists in generation report")
    evidence = prereg.get("evidence_sha256", {})
    source_closure = {
        "collector": args.generator.resolve(),
        "auditor": args.auditor.resolve(),
        "resize_source": args.resize_source.resolve(),
    }
    if any(sha256(path) != evidence.get(name) for name, path in source_closure.items()):
        raise RuntimeError("v455 source closure no longer matches its preregistered evidence SHA")
    bound = {
        "v455_preregistration": prereg_path,
        "v455_generation_report": report_path,
        "v455_legacy_audit": legacy_audit_path,
        "v455_generator": args.generator.resolve(),
        "v455_auditor": args.auditor.resolve(),
        "resize_source": args.resize_source.resolve(),
    }
    payload = {
        "format": "strict-track2-v457-v455-immutable-reconciliation-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "classification": "read-only immutable audit reconciliation; no simulator, model, reward, policy, or RL",
        "v455_registry": str(reg),
        "immutable_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in bound.items()},
        "immutable_npz": [{"name": path.name, "path": str(path.resolve()), "sha256": sha256(path)} for path in npz_paths],
        "original_technical_gate": prereg["technical_gate"],
        "original_branches": prereg["branches"],
        "original_execution": prereg["execution"],
        "original_evidence_sha256": evidence,
        "legacy_check_order": list(EXPECTED_CHECKS),
        "legacy_expected_pass_count": 11,
        "legacy_expected_only_failure": "paired_endpoint_residual_only",
        "canonical_guard_key": "factual_public_endpoint_mae_diagnostic_only",
        "legacy_comparator_alias": "factual_public_fidelity_diagnostic_only",
        "reconciliation_rule": "canonical endpoint-MAE diagnostic-only key is the sole semantic alias for the legacy fidelity diagnostic-only comparator key",
        "guards": {
            "v455_inputs_read_only": True,
            "thresholds_or_formulas_changed": False,
            "public_mae_is_diagnostic_only": True,
            "development_or_final_used": False,
            "reward_loaded": False,
            "simulator_run": False,
            "model_or_policy_updates": 0,
            "rl_authorized": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
