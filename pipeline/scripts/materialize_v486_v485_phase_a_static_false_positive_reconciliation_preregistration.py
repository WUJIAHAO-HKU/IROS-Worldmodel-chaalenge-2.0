#!/usr/bin/env python3
"""Register one read-only v485 static false-positive reconciliation.

Design/source only until the contract's reconciler placeholder is replaced by
an independently reviewed source record.  This program never imports v169,
runs a model, or mutates the frozen b73 lineage.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

CONTRACT_SHA256 = "701236b3e0bfde3eb6e70758891df37370b2b99b151eb31850830b67856049fa"
CONTRACT_FORMAT = "strict-track2-v486-v485-phase-a-static-false-positive-reconciliation-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_reconciler_pending_independent_review_no_execution_authority"
FORMAL_FORMAT = "strict-track2-v486-v485-phase-a-static-false-positive-reconciliation-preregistration-v1"
FORMAL_STATUS = "preregistered_exact_one_readonly_static_reconciliation_authorized"
OLD_FORMAL_FORMAT = "strict-track2-v485-v482-v169-cache-determinism-qualification-preregistration-v1"
OLD_STATIC_FORMAT = "strict-track2-v485-v169-cache-qualification-static-audit-v1"
FALSE_KEYS = {"determinism_scope_exact", "driver_owned_full_lifetime_logs_and_completion", "no_training_reward_outcome"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contains_placeholder(value) -> bool:
    if isinstance(value, str):
        return any(token in value for token in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def regular(path: Path, want_sha: str, want_bytes: int | None = None) -> dict:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"not regular nonsymlink: {path}")
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    if actual_sha != want_sha or (want_bytes is not None and actual_bytes != want_bytes):
        raise RuntimeError(f"file closure mismatch: {path}")
    return {"path": str(path), "sha256": actual_sha, "logical_bytes": actual_bytes}


def verify_record(path: Path, record: dict) -> dict:
    if set(record) not in ({"path", "sha256"}, {"path", "sha256", "logical_bytes"}):
        raise RuntimeError("record keyset")
    if path.resolve() != Path(record["path"]).resolve():
        raise RuntimeError("record path")
    return regular(path, record["sha256"], record.get("logical_bytes"))


def exact_tree(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("tree root")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("tree symlink")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError("tree nonregular")
    lines = "".join(f"{digest}  {rel}\n" for rel, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):
        raise FileExistsError(path)
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def copy_fsync(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256_file(destination) != sha256_file(source) or destination.stat().st_size != source.stat().st_size:
        raise RuntimeError("persistent log copy mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--old-preregistration", type=Path, required=True)
    parser.add_argument("--phase-a-contract", type=Path, required=True)
    parser.add_argument("--old-static-source", type=Path, required=True)
    parser.add_argument("--old-static-receipt", type=Path, required=True)
    parser.add_argument("--old-static-log", type=Path, required=True)
    parser.add_argument("--reconciler-source", type=Path, required=True)
    parser.add_argument("--reconciler-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--registration-root", type=Path, required=True)
    args = parser.parse_args()

    contract_record = regular(args.contract, CONTRACT_SHA256)
    contract = json.loads(args.contract.read_text())
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS:
        raise RuntimeError("contract schema")
    if contains_placeholder(contract):
        raise RuntimeError("contract retains future placeholder")
    if contract.get("current_design_authorization") != {
        "readonly_static_reconciliation_authorized": False,
        "attempts_authorized": 0,
        "phase_a_cache_qualification_authorized": False,
        "cache_reuse_authorized": False,
        "training_authorized": False,
        "folds_authorized": 0,
        "policy_updates": 0,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "rl_authorized": False,
        "submission_authorized": False,
    }:
        raise RuntimeError("design authority")

    parent = contract["frozen_parent"]
    old_formal_record = verify_record(args.old_preregistration, parent["preregistration"])
    phase_a_contract_record = verify_record(args.phase_a_contract, parent["phase_a_design_contract"])
    old_static_source_record = verify_record(args.old_static_source, parent["old_static_source"])
    old_receipt_record = verify_record(args.old_static_receipt, parent["old_failed_static_receipt"])
    volatile_log_record = verify_record(args.old_static_log, parent["old_static_log_volatile_source"])
    reconciler_record = verify_record(args.reconciler_source, contract["future_reconciler_source"])
    materializer_record = regular(args.materializer_source, args.materializer_sha)
    if args.materializer_source.resolve() != Path(__file__).resolve() or args.materializer_sha != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("materializer self closure")

    old_formal = json.loads(args.old_preregistration.read_text())
    phase_a_contract = json.loads(args.phase_a_contract.read_text())
    old_receipt = json.loads(args.old_static_receipt.read_text())
    if old_formal.get("format") != OLD_FORMAL_FORMAT or old_formal.get("status") != "preregistered_cache_qualification_pending_postregistration_static_authority":
        raise RuntimeError("old formal schema")
    if old_formal.get("authorization") != {
        "phase_a_cache_qualification_authorized": False,
        "attempts_authorized": 0,
        "cache_reuse_authorized": False,
        "training_authorized": False,
        "folds_authorized": 0,
        "policy_updates": 0,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "rl_authorized": False,
        "submission_authorized": False,
    } or old_formal.get("runtime_observation") != {"phase_a_executed": False, "training_launched": False, "folds": 0, "policy_updates": 0}:
        raise RuntimeError("old formal authority")
    terminal = contract["old_static_terminal_schema"]
    if old_receipt.get("format") != terminal["format"] or old_receipt.get("status") != terminal["status"] or old_receipt.get("passed") is not False:
        raise RuntimeError("old static terminal")
    checks = old_receipt.get("checks")
    check_keys = sorted(checks) if isinstance(checks, dict) else []
    if len(check_keys) != terminal["check_count"] or old_receipt.get("check_keys") != check_keys or canonical_sha(check_keys) != terminal["check_key_set_sha256"] or old_receipt.get("check_key_set_sha256") != terminal["check_key_set_sha256"] or old_receipt.get("checks_sha256") != canonical_sha(checks):
        raise RuntimeError("old static check schema")
    if {name for name, value in checks.items() if value is False} != FALSE_KEYS or sum(value is True for value in checks.values()) != terminal["true_check_count"]:
        raise RuntimeError("old static 44/3 partition")
    if old_receipt.get("runtime_observation") != terminal["runtime_observation_exact"] or any(old_receipt.get(key) != value for key, value in terminal["top_level_authorization_exact"].items()):
        raise RuntimeError("old static runtime/authority")
    if old_receipt.get("preregistration") != {"path": old_formal_record["path"], "sha256": old_formal_record["sha256"]} or old_receipt.get("contract") != {"path": phase_a_contract_record["path"], "sha256": phase_a_contract_record["sha256"]} or old_receipt.get("static_auditor_self_sha256") != old_static_source_record["sha256"]:
        raise RuntimeError("old static ancestry")

    exact7 = contract["exact7_source_closure"]
    records = []
    for expected in exact7["records"]:
        record = regular(Path(expected["path"]), expected["sha256"], expected["logical_bytes"])
        records.append({"role": expected["role"], **record})
    if [row["role"] for row in records] != exact7["roles_in_order"] or canonical_sha(records) != exact7["canonical_records_digest_sha256"]:
        raise RuntimeError("exact7 digest")
    if old_formal.get("execution_source_records") != records or old_formal.get("execution_sources_digest_sha256") != exact7["canonical_records_digest_sha256"] or old_receipt.get("sources") != old_formal.get("execution_sources") or old_receipt.get("sources_digest_sha256") != exact7["canonical_records_digest_sha256"]:
        raise RuntimeError("exact7 ancestry")
    if phase_a_contract.get("format") != "strict-track2-v485-v482-v169-cache-determinism-qualification-design-contract-v7":
        raise RuntimeError("Phase-A contract format")

    old_tree_spec = parent["old_registration_tree"]
    old_tree = exact_tree(Path(old_tree_spec["root"]))
    for key in ("inventory", "file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"):
        if old_tree[key] != old_tree_spec[key]:
            raise RuntimeError(f"old REG tree: {key}")
    for entry in parent["required_absence"]:
        if os.path.lexists(entry["path"]):
            raise RuntimeError(f"forbidden old output exists: {entry['path']}")

    reconciler_text = args.reconciler_source.read_text()
    reconciler_tree = ast.parse(reconciler_text)
    imported = {alias.name for node in ast.walk(reconciler_tree) if isinstance(node, ast.Import) for alias in node.names} | {node.module or "" for node in ast.walk(reconciler_tree) if isinstance(node, ast.ImportFrom)}
    if any(any(token in name.lower() for token in ("torch", "v169", "wam_pipeline", "transformers", "reward")) for name in imported):
        raise RuntimeError("reconciler forbidden import")
    for required in ("--reconciliation-preregistration", "--reconciliation-contract", "reconciliation_format", "reconciliation_status", "reconciliation_runtime_observation", "reconciliation_only"):
        if required not in reconciler_text:
            raise RuntimeError(f"reconciler schema token: {required}")

    registration_root = args.registration_root.resolve()
    expected_root = Path(contract["lineage"]["registration_root"])
    if not args.registration_root.is_absolute() or registration_root != expected_root or registration_root != args.registration_root or not registration_root.parent.is_dir() or registration_root.parent.is_symlink():
        raise RuntimeError("registration root")
    prep = registration_root.with_name(registration_root.name + ".registration-prep")
    if os.path.lexists(registration_root) or os.path.lexists(prep):
        raise FileExistsError("registration state")

    persistent_log = registration_root / contract["lineage"]["persistent_old_static_log_copy_relative_path"]
    transparent_output = registration_root / contract["lineage"]["transparent_static_receipt_relative_path"]
    sources = {
        "reconciliation_materializer": materializer_record,
        "reconciler": reconciler_record,
    }
    source_records = [{"role": role, **record} for role, record in sources.items()]
    preregistration = {
        "format": FORMAL_FORMAT,
        "status": FORMAL_STATUS,
        "seed": contract["lineage"]["seed"],
        "design_contract": {"path": contract_record["path"], "sha256": contract_record["sha256"]},
        "design_contract_file": contract_record,
        "frozen_parent": parent,
        "exact7_source_closure": exact7,
        "old_static_terminal": {
            "receipt": old_receipt_record,
            "checks_sha256": old_receipt["checks_sha256"],
            "check_keys": check_keys,
            "true_check_names": sorted(name for name, value in checks.items() if value is True),
            "false_check_names": sorted(FALSE_KEYS),
            "runtime_observation": old_receipt["runtime_observation"],
        },
        "false_positive_recomputation_contract": contract["false_positive_recomputation_contract"],
        "execution_sources": sources,
        "execution_source_records": source_records,
        "execution_sources_digest_sha256": canonical_sha(source_records),
        "persistent_old_static_log": {"path": str(persistent_log), "sha256": volatile_log_record["sha256"], "logical_bytes": volatile_log_record["logical_bytes"]},
        "volatile_log_source_at_registration": volatile_log_record,
        "old_input_records": {
            "preregistration": old_formal_record,
            "phase_a_design_contract": phase_a_contract_record,
            "old_static_source": old_static_source_record,
            "old_failed_static_receipt": old_receipt_record,
        },
        "registration_initial_inventory": {
            "exact_relative_paths": [contract["lineage"]["persistent_old_static_log_copy_relative_path"], "preregistration.json"],
            "file_count": 2,
            "persistent_log_sha256": volatile_log_record["sha256"],
            "persistent_log_bytes": volatile_log_record["logical_bytes"],
            "preregistration_self_excluded_from_digest": True,
            "no_other_entries": True,
        },
        "transparent_static_receipt_path": str(transparent_output),
        "transparent_static_receipt_contract": contract["transparent_static_receipt_contract"],
        "authorization": contract["reconciliation_registration_contract"]["authorization_exact"],
        "runtime_observation": contract["reconciliation_registration_contract"]["runtime_observation_at_registration"],
        "old_lineage_immutable": True,
        "qualification_root_absent": True,
        "old_static_retry_authorized": False,
    }

    before = {
        "old_formal": sha256_file(args.old_preregistration),
        "phase_a_contract": sha256_file(args.phase_a_contract),
        "old_static_source": sha256_file(args.old_static_source),
        "old_static_receipt": sha256_file(args.old_static_receipt),
        "old_static_log": sha256_file(args.old_static_log),
        "old_tree": exact_tree(args.old_preregistration.parent),
    }
    prep.mkdir()
    try:
        evidence_dir = prep / "immutable_evidence"
        evidence_dir.mkdir()
        copied_log = evidence_dir / "v485_static_b73.log"
        copy_fsync(args.old_static_log, copied_log)
        fsync_dir(evidence_dir)
        atomic_json(prep / "preregistration.json", preregistration)
        fsync_dir(prep)
        copied_before_promote = regular(copied_log, volatile_log_record["sha256"], volatile_log_record["logical_bytes"])
        volatile_before_promote = verify_record(args.old_static_log, parent["old_static_log_volatile_source"])
        if copied_before_promote["sha256"] != volatile_before_promote["sha256"] or copied_before_promote["logical_bytes"] != volatile_before_promote["logical_bytes"]:
            raise RuntimeError("pre-promote persistent/volatile log mismatch")
        after = {
            "old_formal": sha256_file(args.old_preregistration),
            "phase_a_contract": sha256_file(args.phase_a_contract),
            "old_static_source": sha256_file(args.old_static_source),
            "old_static_receipt": sha256_file(args.old_static_receipt),
            "old_static_log": sha256_file(args.old_static_log),
            "old_tree": exact_tree(args.old_preregistration.parent),
        }
        if before != after or any(os.path.lexists(entry["path"]) for entry in parent["required_absence"]):
            raise RuntimeError("old lineage changed during registration")
        os.replace(prep, registration_root)
        fsync_dir(registration_root.parent)
    except BaseException:
        if prep.exists() and not registration_root.exists():
            shutil.rmtree(prep)
        raise

    formal_path = registration_root / "preregistration.json"
    copied_path = registration_root / contract["lineage"]["persistent_old_static_log_copy_relative_path"]
    if sha256_file(copied_path) != volatile_log_record["sha256"] or copied_path.stat().st_size != volatile_log_record["logical_bytes"] or os.path.lexists(transparent_output):
        raise RuntimeError("postregistration state")
    tree = exact_tree(registration_root)
    if tree["file_count"] != 2 or sorted(row[0] for row in tree["inventory"]) != ["immutable_evidence/v485_static_b73.log", "preregistration.json"]:
        raise RuntimeError("new REG inventory")
    print(json.dumps({"path": str(formal_path), "sha256": sha256_file(formal_path), "registration_root": str(registration_root), "persistent_log": {"path": str(copied_path), "sha256": sha256_file(copied_path), "logical_bytes": copied_path.stat().st_size}, "registration_tree": tree, "transparent_static_receipt_absent": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
