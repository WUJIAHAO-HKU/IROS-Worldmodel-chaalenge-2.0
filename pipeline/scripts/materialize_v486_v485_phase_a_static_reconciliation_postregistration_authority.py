#!/usr/bin/env python3
"""Materialize one postregistration authority for the c71 reconciliation only.

The source is design-only and consumes a frozen reviewed execution wrapper.
It never imports/runs v169, c71, a model, reward code, or training code.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

AUTHORITY_CONTRACT_PATH = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v486_v485_phase_a_static_reconciliation_postregistration_authority_contract.json")
CONTRACT_FORMAT = "strict-track2-v486-v485-phase-a-static-reconciliation-postregistration-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v486-v485-phase-a-static-reconciliation-postregistration-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_readonly_static_reconciliation_attempt"


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pending(value) -> bool:
    if isinstance(value, str):
        return any(token in value for token in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(pending(item) for item in value.values())
    if isinstance(value, list):
        return any(pending(item) for item in value)
    return False


def regular(path: Path | str, want_sha: str, want_bytes: int | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"not regular nonsymlink: {path}")
    actual_sha = sha(path)
    actual_bytes = path.stat().st_size
    if actual_sha != want_sha or (want_bytes is not None and actual_bytes != want_bytes):
        raise RuntimeError(f"file closure mismatch: {path}")
    return {"path": str(path), "sha256": actual_sha, "logical_bytes": actual_bytes}


def verify_record(record: dict) -> dict:
    if set(record) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record keyset")
    return regular(record["path"], record["sha256"], record["logical_bytes"])


def exact_tree(root: Path | str) -> dict:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {rel}\n" for rel, digest, _ in rows).encode()
    compact = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(compact).hexdigest(),
    }


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):
        raise FileExistsError(path)
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def matching_processes() -> list[dict]:
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "reconcile_v486_v485_static_false_positive.py" in command or "materialize_v486_v485_phase_a_static_false_positive_reconciliation_preregistration.py" in command:
            rows.append({"pid": int(proc.name), "command": command})
    return rows


def input_snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, str]) -> dict:
    file_rows = []
    for label, path_text in sorted(files.items()):
        path = Path(path_text)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"snapshot file: {label}")
        file_rows.append({"label": label, "path": str(path.resolve()), "sha256": sha(path), "logical_bytes": path.stat().st_size})
    tree_rows = {label: exact_tree(path) for label, path in sorted(trees.items())}
    absence_rows = []
    for label, path_text in sorted(absences.items()):
        path = Path(path_text)
        absent = not os.path.lexists(path)
        if not absent:
            raise RuntimeError(f"snapshot absence: {label}")
        absence_rows.append({"label": label, "path": str(path.resolve()), "absent": True})
    payload = {"files": file_rows, "trees": tree_rows, "absences": absence_rows}
    return {**payload, "snapshot_sha256": canonical_sha(payload)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--reconciliation-preregistration", type=Path, required=True)
    parser.add_argument("--authority-materializer-source", type=Path, required=True)
    parser.add_argument("--authority-materializer-sha", required=True)
    parser.add_argument("--failure-transcript", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    args = parser.parse_args()

    if args.contract != AUTHORITY_CONTRACT_PATH or args.contract.resolve() != AUTHORITY_CONTRACT_PATH:
        raise RuntimeError("authority contract canonical path")
    contract_record = regular(args.contract, args.contract_sha)
    contract = json.loads(args.contract.read_text())
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS or pending(contract):
        raise RuntimeError("authority contract not frozen")
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

    formal_spec = contract["reconciliation_preregistration"]
    if args.reconciliation_preregistration.resolve() != Path(formal_spec["path"]).resolve():
        raise RuntimeError("formal path")
    formal_record = regular(args.reconciliation_preregistration, formal_spec["sha256"], formal_spec["logical_bytes"])
    formal = json.loads(args.reconciliation_preregistration.read_text())
    if formal.get("format") != formal_spec["format"] or formal.get("status") != formal_spec["status"] or formal.get("authorization") != formal_spec["authorization_exact"] or formal.get("runtime_observation") != formal_spec["runtime_observation_exact"]:
        raise RuntimeError("formal schema/authority")
    if formal.get("design_contract") != {"path": contract["execution_sources"]["reconciliation_design_contract"]["path"], "sha256": contract["execution_sources"]["reconciliation_design_contract"]["sha256"]}:
        raise RuntimeError("formal design contract")
    if formal.get("transparent_static_receipt_path") != contract["lineage"]["transparent_static_receipt_path"]:
        raise RuntimeError("transparent output path")

    execution_source_records = {}
    for role, record in contract["execution_sources"].items():
        execution_source_records[role] = verify_record(record)
    if contract.get("future_execution_wrapper_source") != execution_source_records["execution_wrapper"]:
        raise RuntimeError("future wrapper source alias")
    if formal["execution_sources"].get("reconciliation_materializer") != execution_source_records["reconciliation_registration_materializer"] or formal["execution_sources"].get("reconciler") != execution_source_records["reconciler"]:
        raise RuntimeError("formal execution sources")
    if canonical_sha(formal["execution_source_records"]) != formal["execution_sources_digest_sha256"]:
        raise RuntimeError("formal execution source digest")

    authority_materializer_record = regular(args.authority_materializer_source, args.authority_materializer_sha)
    if args.authority_materializer_source.resolve() != Path(__file__).resolve() or args.authority_materializer_sha != sha(Path(__file__).resolve()):
        raise RuntimeError("authority materializer self closure")
    if contract.get("authority_materializer_source") != authority_materializer_record:
        raise RuntimeError("authority materializer contract binding")
    failure_spec = contract["materialization_transport_forensics"]["first_failed_invocation"]["persistent_review_copy"]
    if args.failure_transcript.resolve() != Path(failure_spec["path"]).resolve():
        raise RuntimeError("failure transcript path")
    failure_transcript_record = regular(args.failure_transcript, failure_spec["sha256"], failure_spec["logical_bytes"])

    new_tree_spec = contract["reconciliation_registration_tree"]
    new_tree = exact_tree(new_tree_spec["root"])
    for key in ("inventory", "file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"):
        if new_tree[key] != new_tree_spec[key]:
            raise RuntimeError(f"reconciliation REG tree: {key}")
    old = contract["old_b73_ancestry"]
    old_records = {}
    for role in ("preregistration", "phase_a_design_contract", "old_static_source", "failed_static_receipt", "volatile_static_log", "persistent_static_log"):
        old_records[role] = verify_record(old[role])
    old_tree = exact_tree(old["registration_tree"]["root"])
    for key in ("file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"):
        if old_tree[key] != old["registration_tree"][key]:
            raise RuntimeError(f"old b73 tree: {key}")
    old_receipt = json.loads(Path(old["failed_static_receipt"]["path"]).read_text())
    old_checks = old_receipt.get("checks", {})
    if len(old_checks) != 47 or {name for name, value in old_checks.items() if value is False} != set(old["failed_checks_exact"]) or sum(value is True for value in old_checks.values()) != old["true_check_count"] or old_receipt.get("check_key_set_sha256") != old["check_key_set_sha256"]:
        raise RuntimeError("old static 44/3")

    exact7 = formal["exact7_source_closure"]
    if exact7.get("canonical_records_digest_sha256") != contract["exact7_source_closure"]["records_digest_sha256"] or canonical_sha(exact7["records"]) != exact7["canonical_records_digest_sha256"]:
        raise RuntimeError("exact7 digest")
    for record in exact7["records"]:
        verify_record({key: record[key] for key in ("path", "sha256", "logical_bytes")})

    transport = contract["materialization_transport_forensics"]
    success = transport["successful_invocation"]
    expected_stdout = {
        "path": formal_record["path"],
        "sha256": formal_record["sha256"],
        "registration_root": new_tree_spec["root"],
        "persistent_log": old_records["persistent_static_log"],
        "registration_tree": new_tree,
        "transparent_static_receipt_absent": True,
    }
    stdout_bytes = (json.dumps(expected_stdout, sort_keys=True) + "\n").encode()
    if hashlib.sha256(stdout_bytes).hexdigest() != success["tool_captured_stdout"]["sha256"] or len(stdout_bytes) != success["tool_captured_stdout"]["logical_bytes"] or success["tool_captured_stdout"].get("trailing_lf") is not True:
        raise RuntimeError("successful stdout forensic")
    if success["tool_captured_stderr"] != {"sha256": hashlib.sha256(b"").hexdigest(), "logical_bytes": 0} or any(success[key] is not False for key in ("native_argv_persistent_log_available", "native_stdout_persistent_log_available", "native_stderr_persistent_log_available")):
        raise RuntimeError("successful unavailable/empty evidence")
    if transport["first_failed_invocation"]["classification"] != "reconstructed_non_native_tool_console_capture" or transport["first_failed_invocation"]["stderr_native_persistent_log_available"] is not False:
        raise RuntimeError("failure forensic disclosure")

    authority_root = args.authority_root.resolve()
    if authority_root != Path(contract["lineage"]["authority_root"]) or args.authority_root != authority_root or not authority_root.parent.is_dir() or authority_root.parent.is_symlink():
        raise RuntimeError("authority root")
    prep = authority_root.with_name(authority_root.name + ".registration-prep")
    if os.path.lexists(authority_root) or os.path.lexists(prep):
        raise FileExistsError("authority registration state")
    for entry in contract["historical_pre_materialization_absences"].values():
        if os.path.lexists(entry["path"]):
            raise RuntimeError(f"required absence: {entry['path']}")
    if matching_processes():
        raise RuntimeError("reconciliation process already running")

    wrapper_text = Path(execution_source_records["execution_wrapper"]["path"]).read_text()
    wrapper_tree = ast.parse(wrapper_text)
    for token in ("intent.json", "reconciler_stdout.log", "reconciler_stderr.log", "terminal_receipt.json", "transparent_static_receipt_path", "attempts_authorized", "retry_authorized"):
        if token not in wrapper_text:
            raise RuntimeError(f"wrapper schema token: {token}")
    imported = {alias.name for node in ast.walk(wrapper_tree) if isinstance(node, ast.Import) for alias in node.names} | {node.module or "" for node in ast.walk(wrapper_tree) if isinstance(node, ast.ImportFrom)}
    if any(any(token in name.lower() for token in ("torch", "v169", "wam_pipeline", "transformers", "reward")) for name in imported):
        raise RuntimeError("wrapper forbidden import")

    file_inputs = {
        "authority_design_contract": str(args.contract),
        "reconciliation_preregistration": str(args.reconciliation_preregistration),
        "authority_materializer": str(args.authority_materializer_source),
        "failure_transcript": str(args.failure_transcript),
    }
    for role, record in execution_source_records.items():
        file_inputs[f"execution_source::{role}"] = record["path"]
    for role, record in old_records.items():
        file_inputs[f"old_b73::{role}"] = record["path"]
    for index, record in enumerate(exact7["records"]):
        file_inputs[f"exact7::{index:02d}::{record['role']}"] = record["path"]
    tree_inputs = {"reconciliation_registration": new_tree_spec["root"], "old_b73_registration": old["registration_tree"]["root"]}
    absence_inputs = {f"historical::{name}": item["path"] for name, item in sorted(contract["historical_pre_materialization_absences"].items())}
    absence_inputs["authority_registration_prep"] = str(prep)
    pre_snapshot = input_snapshot(file_inputs, tree_inputs, absence_inputs)
    post_snapshot = input_snapshot(file_inputs, tree_inputs, absence_inputs)
    if pre_snapshot != post_snapshot:
        raise RuntimeError("authority input snapshot drift")
    stable_absence_inputs = {key: value for key, value in absence_inputs.items() if key != "authority_registration_prep"}
    stable_pre_snapshot = input_snapshot(file_inputs, tree_inputs, stable_absence_inputs)

    checks = {
        "formal_exact_nonauthorizing_phase_a": True,
        "reconciliation_REG_exact2": True,
        "old_b73_REG_and_44_3_exact": True,
        "exact7_current": True,
        "persistent_and_volatile_log_exact": True,
        "first_failure_disclosed_non_native_reconstructed": True,
        "successful_transport_stdout_recomputed": True,
        "successful_native_logs_explicitly_unavailable": True,
        "reconciler_and_wrapper_source_exact": True,
        "all_required_outputs_absent": True,
        "no_reconciliation_process": True,
        "input_pre_post_snapshots_equal": True,
    }
    receipt = {
        "format": OUTPUT_FORMAT,
        "status": OUTPUT_STATUS,
        "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": authority_materializer_record,
        "reconciliation_preregistration": formal_record,
        "reconciliation_design_contract": execution_source_records["reconciliation_design_contract"],
        "reconciliation_registration_materializer": execution_source_records["reconciliation_registration_materializer"],
        "reconciler_source": execution_source_records["reconciler"],
        "execution_wrapper_source": execution_source_records["execution_wrapper"],
        "old_parent_ancestry": {"records": old_records, "failed_checks": old["failed_checks_exact"], "true_check_count": old["true_check_count"]},
        "exact7_source_closure": {"records": exact7["records"], "records_digest_sha256": exact7["canonical_records_digest_sha256"]},
        "old_b73_tree": old_tree,
        "reconciliation_REG_exact2": new_tree,
        "transport_forensics": {
            **transport,
            "first_failed_invocation": {**transport["first_failed_invocation"], "persistent_review_copy": failure_transcript_record},
            "successful_materializer_invocation_persistent_argv_log_available": False,
            "successful_materializer_invocation_transport_disclosed": True,
        },
        "attempt_root": contract["lineage"]["attempt_root"],
        "transparent_static_receipt_path": contract["lineage"]["transparent_static_receipt_path"],
        "historical_pre_materialization_absences": {name: {**item, "absent": True} for name, item in sorted(contract["historical_pre_materialization_absences"].items())},
        "required_absences": {name: {**item, "absent": True} for name, item in sorted(contract["required_current_absences_after_authority"].items())},
        "checks": checks,
        "input_pre_snapshot": pre_snapshot,
        "input_post_snapshot": post_snapshot,
        "input_snapshots_exactly_equal": True,
        "authorization": contract["authority_receipt_contract"]["authorization_exact"],
        "runtime_observation": contract["authority_receipt_contract"]["runtime_observation_exact"],
    }
    if set(receipt) != set(contract["authority_receipt_contract"]["required_top_level_keys"]):
        raise RuntimeError("authority receipt keyset")

    prep.mkdir()
    try:
        atomic_json(prep / contract["lineage"]["authority_receipt_relative_path"], receipt)
        fsync_dir(prep)
        stable_post_snapshot = input_snapshot(file_inputs, tree_inputs, stable_absence_inputs)
        if stable_post_snapshot != stable_pre_snapshot:
            raise RuntimeError("authority inputs changed before promote")
        os.replace(prep, authority_root)
        fsync_dir(authority_root.parent)
    except BaseException:
        if prep.exists() and not authority_root.exists():
            shutil.rmtree(prep)
        raise
    output = authority_root / contract["lineage"]["authority_receipt_relative_path"]
    print(json.dumps({"path": str(output), "sha256": sha(output), "authority_root": str(authority_root), "reconciliation_executed": False, "phase_a_cache_qualification_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
