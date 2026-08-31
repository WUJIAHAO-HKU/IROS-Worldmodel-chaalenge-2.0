#!/usr/bin/env python3
"""Rerun frozen v455 audit and reconcile its one misspelled guard lookup."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "recomputed-output", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.preregistration.read_text())
    if pre.get("format") != "strict-track2-v457-v455-immutable-reconciliation-preregistration-v1":
        raise RuntimeError("unexpected v457 preregistration")
    immutable = pre["immutable_files"]
    immutable_npz = pre["immutable_npz"]
    hash_checks = {
        name: Path(spec["path"]).is_file() and sha256(Path(spec["path"])) == spec["sha256"]
        for name, spec in immutable.items()
    }
    hash_checks["four_npz"] = len(immutable_npz) == 4 and all(
        Path(spec["path"]).is_file() and sha256(Path(spec["path"])) == spec["sha256"]
        for spec in immutable_npz
    )
    v455_pre_path = Path(immutable["v455_preregistration"]["path"])
    v455_report_path = Path(immutable["v455_generation_report"]["path"])
    v455_legacy_path = Path(immutable["v455_legacy_audit"]["path"])
    v455_generator = Path(immutable["v455_generator"]["path"])
    v455_auditor = Path(immutable["v455_auditor"]["path"])
    resize_source = Path(immutable["resize_source"]["path"])
    dataset_dir = v455_report_path.parent
    if args.recomputed_output.exists():
        raise RuntimeError("v457 recomputed output already exists")
    command = [
        sys.executable, str(v455_auditor),
        "--preregistration", str(v455_pre_path),
        "--dataset-dir", str(dataset_dir),
        "--generator", str(v455_generator),
        "--resize-source", str(resize_source),
        "--output", str(args.recomputed_output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=60)
    legacy = json.loads(v455_legacy_path.read_text())
    recomputed = json.loads(args.recomputed_output.read_text())
    v455_pre = json.loads(v455_pre_path.read_text())
    v455_report = json.loads(v455_report_path.read_text())
    legacy_auditor_source = v455_auditor.read_text()
    legacy_checks = legacy["checks"]
    expected_order = pre["legacy_check_order"]
    only_failures = [name for name, value in legacy_checks.items() if value is not True]
    immutable_recompute = canonical(recomputed) == canonical(legacy)
    row_metrics_bitwise = canonical(recomputed.get("rows")) == canonical(legacy.get("rows"))
    report_guards = v455_report.get("guards", {})
    pre_guards = v455_pre.get("guards", {})
    alias_valid = (
        pre["canonical_guard_key"] == "factual_public_endpoint_mae_diagnostic_only"
        and pre["legacy_comparator_alias"] == "factual_public_fidelity_diagnostic_only"
        and report_guards.get("factual_public_endpoint_mae_diagnostic_only") is True
        and "factual_public_fidelity_diagnostic_only" not in report_guards
        and pre_guards.get("paired_endpoint_residual_only") is True
        and report_guards.get("paired_endpoint_residual_only") is True
        and report_guards.get("rl_authorized") is False
        and "paired endpoint residual parent data only" in v455_pre.get("classification", "")
    )
    corrected_checks = dict(legacy_checks)
    corrected_checks["paired_endpoint_residual_only"] = alias_valid
    semantic_differences = {
        name: (legacy_checks[name], corrected_checks[name])
        for name in legacy_checks if legacy_checks[name] != corrected_checks[name]
    }
    checks = {
        "all_immutable_hashes": all(hash_checks.values()),
        "original_auditor_expected_exit2": completed.returncode == 2,
        "full_original_audit_bitwise_recomputed": immutable_recompute,
        "all_row_metrics_bitwise_identical": row_metrics_bitwise,
        "legacy_exact_11_of_12": list(legacy_checks) == expected_order and sum(value is True for value in legacy_checks.values()) == pre["legacy_expected_pass_count"],
        "legacy_only_failure_is_known_key": only_failures == [pre["legacy_expected_only_failure"]],
        "canonical_alias_is_true": alias_valid,
        "legacy_bad_lookup_exactly_once": legacy_auditor_source.count('guards.get("factual_public_fidelity_diagnostic_only") is True') == 1,
        "corrected_only_one_boolean": semantic_differences == {"paired_endpoint_residual_only": (False, True)} and set(corrected_checks) == set(legacy_checks),
        "all_original_thresholds_and_formulas_bound": pre["original_technical_gate"] == v455_pre["technical_gate"] and pre["original_branches"] == v455_pre["branches"] and pre["original_execution"] == v455_pre["execution"],
        "all_original_source_evidence_bound": pre["original_evidence_sha256"] == v455_pre["evidence_sha256"] and v455_pre["evidence_sha256"]["collector"] == sha256(v455_generator) and v455_pre["evidence_sha256"]["auditor"] == sha256(v455_auditor) and v455_pre["evidence_sha256"]["resize_source"] == sha256(resize_source),
        "all_corrected_checks_pass": all(corrected_checks.values()),
        "no_public_mae_pass_threshold": v455_pre["technical_gate"].get("factual_public_endpoint_mae") == "diagnostic_only",
        "generation_was_passed": v455_report.get("passed") is True,
    }
    passed = all(checks.values())
    receipt = {
        "format": "strict-track2-v457-v455-immutable-reconciliation-receipt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
        "hash_checks": hash_checks,
        "legacy_checks": legacy_checks,
        "corrected_checks": corrected_checks,
        "legacy_failed_only_due_to_key_alias": passed,
        "recomputed_v455_audit_sha256": sha256(args.recomputed_output),
        "endpoint_parent_data_authorized": passed,
        "authorization_scope": "public-train within-simulator paired endpoint residual parent expansion only",
        "guards": {
            "v455_inputs_modified": False,
            "thresholds_or_formulas_changed": False,
            "public_mae_is_diagnostic_only": True,
            "simulator_run": False,
            "reward_loaded": False,
            "model_or_policy_updates": 0,
            "rl_authorized": False,
            "real_submission": False,
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 2

if __name__ == "__main__":
    raise SystemExit(main())
