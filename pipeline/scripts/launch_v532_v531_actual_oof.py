#!/usr/bin/env python3
"""Sole launcher for the v532 repaired actual-OOF execution boundary."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

SEED = 1666
ACTIVE_SOURCE_ROLE_ORDER = [
    "authority_design_contract", "authority_materializer", "actual_oof_execution_preregistration",
    "actual_oof_execution_manifest", "actual_oof_executor", "actual_oof_auditor", "actual_oof_launcher",
]
ACTIVE_SOURCE_PATHS = {
    "authority_design_contract": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_contract.json",
    "authority_materializer": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority.py",
    "actual_oof_execution_preregistration": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_execution_preregistration.json",
    "actual_oof_execution_manifest": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_execution_manifest.json",
    "actual_oof_executor": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/execute_v532_v531_actual_oof.py",
    "actual_oof_auditor": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/audit_v532_v531_actual_oof.py",
    "actual_oof_launcher": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v532_v531_actual_oof.py",
}
SOURCE_ROLE_ORDER = [
    "authority_materializer", "actual_oof_execution_preregistration", "actual_oof_execution_manifest",
    "actual_oof_executor", "actual_oof_auditor", "actual_oof_launcher", "v531_failure_forensic",
    "v531_invalid_authority_design_contract", "v531_invalid_authority_materializer",
    "v531_invalid_actual_oof_execution_preregistration", "v531_invalid_actual_oof_execution_manifest",
    "v531_invalid_actual_oof_executor", "v531_invalid_actual_oof_auditor", "v531_invalid_actual_oof_launcher",
    "v531_unconsumed_authority_receipt", "v530_external_terminal_process", "v527_candidate_receipt",
    "v527_authority_receipt", "v528_deployment_receipt", "v524_qualification_terminal",
    "v524_qualification_report", "v524_qualification_audit", "v524_cache_manifest", "v482_preregistration",
    "v482_runtime_source", "v482_trainer_source", "v482_model_design_contract", "v482_s0_auditor", "v478_selection",
]
SOURCE_ALIASES = {role: f"{role}_source" for role in SOURCE_ROLE_ORDER}
SOURCE_ALIASES.update({
    "authority_design_contract": "authority_design_contract",
    "v482_runtime_source": "v482_runtime_source_source",
    "v482_trainer_source": "v482_trainer_source_source",
})
EXPECTED_LINEAGE = {
    "canonical_v532_execution_not_performed": True,
    "fresh_actual_oof_authority_only": True,
    "v531_failed_attempt_or_output_not_published": True,
    "v531_authority_durable_consumption_receipt_available": False,
    "v531_authority_consumed_field_zero": True,
    "v531_physical_launcher_executor_entry_recorded_no_retry": True,
    "v531_failed_lineage_retry_authorized": False,
    "v531_failure_forensic_current": True,
    "v524_qualification_exact20_readonly": True,
    "v482_dataset_model_source_current_revalidated": True,
}
PREREG_KEYS = {
    "format", "status", "seed", "classification", "authority_contract_path", "fresh_authority_root",
    "fresh_authority_prep_root", "fresh_attempt_root", "fresh_attempt_prep_root", "fresh_oof_root",
    "fresh_oof_prep_root", "execution_manifest", "active_source_role_order", "active_source_paths",
    "input_contract", "output_contract", "authorization", "boundary_partition", "required_environment",
    "service_health", "historical_training_authority_not_inherited", "no_space_transport_required",
    "noncyclic_transport_deployment_record_required",
}
MANIFEST_KEYS = {
    "format", "status", "seed", "active_source_role_order", "active_source_paths", "fold_order", "branches",
    "selection_count", "ordered_oof_rows_count", "fold_row_counts", "ordering", "ordered_rows",
    "ordered_rows_canonical_sha256", "cache_input", "dataset_input", "model_and_source_input", "output_schema",
    "execution_contract",
}


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def json_exact(left, right) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == json.dumps(right, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    if path.is_symlink() or not path.is_file() or sha(path) != digest:
        raise RuntimeError(f"source drift:{path}")


def validate_active_records(contract: dict, authority: dict, args) -> None:
    shared_roles = ACTIVE_SOURCE_ROLE_ORDER[1:]
    contract_records = contract.get("active_source_records")
    authority_records = authority.get("active_source_records")
    receipt_contract = contract.get("authority_receipt_contract", {})
    if (not isinstance(contract_records, dict) or set(contract_records) != set(shared_roles)
            or not isinstance(authority_records, dict) or set(authority_records) != set(ACTIVE_SOURCE_ROLE_ORDER)
            or type(receipt_contract.get("contract_active_source_records_count")) is not int
            or receipt_contract["contract_active_source_records_count"] != 6
            or receipt_contract.get("contract_design_record_excluded_to_avoid_self_hash_cycle") is not True
            or type(receipt_contract.get("authority_active_source_records_count")) is not int
            or receipt_contract["authority_active_source_records_count"] != 7
            or receipt_contract.get("authority_design_record_included") is not True):
        raise RuntimeError("active source record schema")
    for role in ACTIVE_SOURCE_ROLE_ORDER:
        record = authority_records[role]
        if (set(record) != {"path", "sha256", "logical_bytes"}
                or record["path"] != ACTIVE_SOURCE_PATHS[role]
                or type(record["logical_bytes"]) is not int
                or authority.get("source_closure", {}).get(role) != record):
            raise RuntimeError(f"authority active source record:{role}")
        if role in shared_roles and (contract_records[role] != record or contract.get("source_closure", {}).get(role) != record):
            raise RuntimeError(f"shared active source record:{role}")
    expected_design = {"path": str(args.contract), "sha256": args.contract_sha,
                       "logical_bytes": args.contract.stat().st_size}
    if authority_records["authority_design_contract"] != expected_design:
        raise RuntimeError("authority design source record")


def validate_source_closures(contract: dict, authority: dict) -> None:
    expected_authority_roles = ["authority_design_contract", *SOURCE_ROLE_ORDER]
    contract_closure = contract.get("source_closure")
    authority_closure = authority.get("source_closure")
    if (contract.get("source_role_order") != SOURCE_ROLE_ORDER
            or not json_exact(contract.get("source_aliases"), SOURCE_ALIASES)
            or not isinstance(contract_closure, dict)
            or set(contract_closure) != set(SOURCE_ROLE_ORDER)
            or canonical_sha(contract_closure) != contract.get("source_closure_sha256")):
        raise RuntimeError("contract source closure")
    if (authority.get("source_role_order") != expected_authority_roles
            or not json_exact(authority.get("source_aliases"), SOURCE_ALIASES)
            or not isinstance(authority_closure, dict)
            or set(authority_closure) != set(expected_authority_roles)
            or canonical_sha(authority_closure) != authority.get("source_closure_sha256")):
        raise RuntimeError("authority source closure")
    for role in SOURCE_ROLE_ORDER:
        if authority_closure[role] != contract_closure[role]:
            raise RuntimeError(f"shared source closure:{role}")
    for role in ACTIVE_SOURCE_ROLE_ORDER[1:]:
        if contract.get(SOURCE_ALIASES[role]) != contract_closure[role]:
            raise RuntimeError(f"contract source alias:{role}")
    for role in expected_authority_roles:
        alias = SOURCE_ALIASES[role]
        if authority.get(alias) != authority_closure[role]:
            raise RuntimeError(f"authority source alias:{role}")
    if not json_exact(contract.get("lineage"), EXPECTED_LINEAGE) or not json_exact(authority.get("lineage"), EXPECTED_LINEAGE):
        raise RuntimeError("source lineage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--authority-receipt", type=Path, required=True); parser.add_argument("--authority-receipt-sha", required=True)
    parser.add_argument("--executor-source", type=Path, required=True); parser.add_argument("--executor-sha", required=True)
    parser.add_argument("--auditor-source", type=Path, required=True); parser.add_argument("--auditor-sha", required=True)
    parser.add_argument("--launcher-sha", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def run(argv=None):
    literal = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(literal)
    if type(args.seed) is not int or args.seed != SEED:
        raise RuntimeError("seed strict int")
    launcher = Path(__file__).resolve()
    exact(launcher, args.launcher_sha); exact(args.executor_source, args.executor_sha); exact(args.auditor_source, args.auditor_sha)
    exact(args.preregistration, args.preregistration_sha); exact(args.manifest, args.manifest_sha)
    exact(args.contract, args.contract_sha); exact(args.authority_receipt, args.authority_receipt_sha)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    authority = json.loads(args.authority_receipt.read_text(encoding="utf-8"))
    if (set(prereg) != PREREG_KEYS
            or set(manifest) != MANIFEST_KEYS
            or prereg.get("format") != "strict-track2-v532-v531-actual-oof-execution-preregistration-v1"
            or manifest.get("format") != "strict-track2-v532-v531-actual-oof-execution-manifest-v1"
            or prereg.get("seed") != SEED
            or manifest.get("seed") != SEED
            or prereg.get("active_source_role_order") != ACTIVE_SOURCE_ROLE_ORDER
            or manifest.get("active_source_role_order") != ACTIVE_SOURCE_ROLE_ORDER
            or contract.get("active_source_role_order") != ACTIVE_SOURCE_ROLE_ORDER
            or authority.get("active_source_role_order") != ACTIVE_SOURCE_ROLE_ORDER
            or prereg.get("active_source_paths") != ACTIVE_SOURCE_PATHS
            or manifest.get("active_source_paths") != ACTIVE_SOURCE_PATHS
            or contract.get("active_source_paths") != ACTIVE_SOURCE_PATHS
            or authority.get("active_source_paths") != ACTIVE_SOURCE_PATHS
            or args.contract != Path(ACTIVE_SOURCE_PATHS["authority_design_contract"])
            or args.preregistration != Path(ACTIVE_SOURCE_PATHS["actual_oof_execution_preregistration"])
            or args.manifest != Path(ACTIVE_SOURCE_PATHS["actual_oof_execution_manifest"])
            or args.executor_source.resolve() != Path(ACTIVE_SOURCE_PATHS["actual_oof_executor"])
            or args.auditor_source.resolve() != Path(ACTIVE_SOURCE_PATHS["actual_oof_auditor"])
            or launcher != Path(ACTIVE_SOURCE_PATHS["actual_oof_launcher"])):
        raise RuntimeError("preregistered active paths")
    validate_source_closures(contract, authority)
    validate_active_records(contract, authority, args)
    if args.output_root != Path(prereg["fresh_oof_root"]):
        raise RuntimeError("fresh output root")
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        spec = importlib.util.spec_from_file_location("v532_actual_oof_executor", args.executor_source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        inner = [
            "--preregistration", str(args.preregistration), "--preregistration-sha", args.preregistration_sha,
            "--manifest", str(args.manifest), "--manifest-sha", args.manifest_sha,
            "--contract", str(args.contract), "--contract-sha", args.contract_sha,
            "--authority-receipt", str(args.authority_receipt), "--authority-receipt-sha", args.authority_receipt_sha,
            "--executor-sha", args.executor_sha, "--auditor-source", str(args.auditor_source), "--auditor-sha", args.auditor_sha,
            "--launcher-source", str(launcher), "--launcher-sha", args.launcher_sha,
            "--launcher-argv-json", json.dumps(literal, separators=(",", ":")),
            "--launcher-pid", str(os.getpid()), "--launcher-ppid", str(os.getppid()),
            "--launcher-pgid", str(os.getpgid(0)), "--launcher-sid", str(os.getsid(0)),
            "--seed", str(args.seed), "--output-root", str(args.output_root),
        ]
        receipt = module.main(inner)
        if receipt.get("passed") is not True or receipt.get("actual_oof_execution_boundary_invocations") != 1 or receipt.get("launcher_invocations") != 1 or receipt.get("executor_invocations") != 1 or receipt.get("auditor_import_invocations") != 1:
            raise RuntimeError("actual OOF boundary result")
        return receipt
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def synthetic_self_test() -> bool:
    parser = build_parser()
    names = {action.dest for action in parser._actions}
    return names == {"help", "preregistration", "preregistration_sha", "manifest", "manifest_sha", "contract", "contract_sha", "authority_receipt", "authority_receipt_sha", "executor_source", "executor_sha", "auditor_source", "auditor_sha", "launcher_sha", "seed", "output_root"}


if __name__ == "__main__":
    if sys.argv[1:] == ["--synthetic-self-test"]:
        print(json.dumps({"passed": synthetic_self_test()}, sort_keys=True)); raise SystemExit(0)
    try:
        result = run(); print(json.dumps({"passed": True, "status": result["status"], "events": result["events"], "folds": result["folds"]}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"passed": False, "status": "failed_no_retry", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr); raise SystemExit(79)
