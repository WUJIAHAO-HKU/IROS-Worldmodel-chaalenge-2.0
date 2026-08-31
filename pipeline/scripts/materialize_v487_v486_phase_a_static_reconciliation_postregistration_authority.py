#!/usr/bin/env python3
"""Materialize a fresh v487 authority receipt; never run reconciliation or Phase-A."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
CONTRACT_PATH = ROOT / "pipeline/scripts/v487_v486_phase_a_static_reconciliation_postregistration_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v487-v486-phase-a-static-reconciliation-postregistration-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_awaiting_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v487-v486-phase-a-static-reconciliation-postregistration-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_fresh_readonly_static_reconciliation_attempt"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pending(value) -> bool:
    if isinstance(value, str):
        return any(token in value for token in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(pending(item) for item in value.values())
    if isinstance(value, list):
        return any(pending(item) for item in value)
    return False


def regular(path: Path | str, digest: str, size: int) -> dict:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular canonical file: {path}")
    if sha(path) != digest or path.stat().st_size != size:
        raise RuntimeError(f"file closure: {path}")
    return {"path": str(path), "sha256": digest, "logical_bytes": size}


def record(value: dict) -> dict:
    if set(value) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record schema")
    return regular(value["path"], value["sha256"], value["logical_bytes"])


def tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or not root.is_dir() or root.is_symlink():
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
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def exact_tree(spec: dict) -> dict:
    observed = tree(spec["root"])
    if observed != {key: spec[key] for key in observed}:
        raise RuntimeError(f"tree closure: {spec['root']}")
    return observed


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


def snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, str]) -> dict:
    file_rows = []
    for label, text in sorted(files.items()):
        path = Path(text)
        if path != path.resolve() or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"snapshot file: {label}")
        file_rows.append({"label": label, "path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size})
    tree_rows = {label: tree(path) for label, path in sorted(trees.items())}
    absence_rows = []
    for label, text in sorted(absences.items()):
        path = Path(text)
        if os.path.lexists(path):
            raise RuntimeError(f"snapshot absence: {label}")
        absence_rows.append({"label": label, "path": str(path), "absent": True})
    body = {"files": file_rows, "trees": tree_rows, "absences": absence_rows}
    return {**body, "snapshot_sha256": csha(body)}


def live_relevant_processes() -> list[int]:
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "reconcile_v486_v485_static_false_positive.py" in cmd or "launch_v487_v486_v485_static_false_positive_reconciliation.py" in cmd:
            hits.append(int(proc.name))
    return hits


def atomic_promotion_fixture() -> None:
    base = Path(tempfile.mkdtemp(prefix="v487-authority-promote-fixture-", dir="/dev/shm"))
    prep = base / "authority.registration-prep"
    final = base / "authority"
    source = base / "immutable-input.bin"
    try:
        with source.open("xb") as stream:
            stream.write(b"v487-authority-promotion-fixture-v1\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_dir(base)
        source_before = {"sha256": sha(source), "logical_bytes": source.stat().st_size}
        prep.mkdir()
        atomic_json(prep / "authority_receipt.json", {"fixture": True})
        fsync_dir(prep)
        if {"sha256": sha(source), "logical_bytes": source.stat().st_size} != source_before:
            raise RuntimeError("fixture input drift")
        os.replace(prep, final)
        fsync_dir(base)
        observed = tree(final)
        if observed["file_count"] != 1 or observed["inventory"][0][0] != "authority_receipt.json":
            raise RuntimeError("fixture promotion tree")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract canonical path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS or pending(contract):
        raise RuntimeError("contract frozen schema")
    if contract.get("current_design_authorization") != {
        "authority_materialization_authorized": False,
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
        "reward_read_authorized": False,
        "dev_hidden_final_outcome_read_authorized": False,
    }:
        raise RuntimeError("design authorization")
    if contract.get("materialization_policy") != {
        "superseded_v486_attempts_authorized": 1,
        "superseded_v486_attempts_consumed": 1,
        "superseded_v486_retry_authorized": False,
        "fresh_v487_attempts_authorized": 1,
        "fresh_v487_attempts_consumed": 0,
        "fresh_v487_retry_authorized": False,
        "transcript_deployment_must_precede_fresh_materializer": True,
        "old_authority_root_reuse_authorized": False,
    }:
        raise RuntimeError("materialization policy")
    materializer_record = regular(args.materializer_source, args.materializer_sha, args.materializer_source.stat().st_size)
    if args.materializer_source.resolve() != Path(__file__).resolve() or materializer_record != contract["authority_materializer_source"]:
        raise RuntimeError("materializer binding")

    formal_record = record(contract["reconciliation_preregistration"])
    formal = json.loads(Path(formal_record["path"]).read_text())
    if formal.get("format") != contract["formal_schema"]["format"] or formal.get("status") != contract["formal_schema"]["status"]:
        raise RuntimeError("formal schema")
    if formal.get("authorization") != contract["formal_schema"]["authorization_exact"] or formal.get("runtime_observation") != contract["formal_schema"]["runtime_observation_exact"]:
        raise RuntimeError("formal authority")
    if formal.get("transparent_static_receipt_path") != contract["lineage"]["transparent_static_receipt_path"]:
        raise RuntimeError("formal output")

    source_records = {name: record(value) for name, value in contract["execution_sources"].items()}
    if contract["future_execution_wrapper_source"] != source_records["execution_wrapper"]:
        raise RuntimeError("wrapper alias")
    if formal.get("design_contract") != {key: source_records["reconciliation_design_contract"][key] for key in ("path", "sha256")}:
        raise RuntimeError("formal design contract")
    if formal.get("execution_sources", {}).get("reconciler") != source_records["reconciler"] or formal.get("execution_sources", {}).get("reconciliation_materializer") != source_records["reconciliation_registration_materializer"]:
        raise RuntimeError("formal execution sources")

    reg_tree = exact_tree(contract["reconciliation_registration_tree"])
    old_tree = exact_tree(contract["old_b73_registration_tree"])
    failure_tree = exact_tree(contract["superseded_authority_materialization"]["evidence_tree"])
    superseded_source_records = {
        name: record(contract["superseded_authority_materialization"][name])
        for name in ("authority_design_contract", "authority_materializer_source", "execution_wrapper_source")
    }
    failure_receipt_record = record(contract["superseded_authority_materialization"]["process_receipt"])
    failure_receipt = json.loads(Path(failure_receipt_record["path"]).read_text())
    if set(failure_receipt) != {"format", "returncode", "wall_seconds", "argv_evidence", "stdout", "stderr", "materializer_invocations"}:
        raise RuntimeError("failure receipt keyset")
    if failure_receipt.get("format") != "strict-track2-v486-authority-materializer-process-receipt-v1" or failure_receipt.get("returncode") != 1 or failure_receipt.get("materializer_invocations") != 1:
        raise RuntimeError("failure receipt facts")
    if not isinstance(failure_receipt.get("wall_seconds"), float) or failure_receipt["wall_seconds"] != 0.11483402:
        raise RuntimeError("failure wall")
    stderr_text = Path(contract["superseded_authority_materialization"]["stderr"]["path"]).read_text(errors="replace")
    if "v486_materializer_failed_invocation_console_capture_reconstructed.txt" not in stderr_text or not any(token in stderr_text for token in ("not regular", "FileNotFoundError", "file closure")):
        raise RuntimeError("failure root cause")
    receipt_text = json.dumps(failure_receipt, sort_keys=True)
    for expected_digest in (
        "64856764178684495674e5e5ed9e4865ad43511b91f3843231547181c1f34ef1",
        "1b1506989e4c0d64e2062eaedab180763bd9adbde069f4a2c6bd70a95806ba4d",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ):
        if expected_digest not in receipt_text:
            raise RuntimeError("failure receipt crosslink")
    transcript = record(contract["deployed_transcript"])

    exact7 = formal.get("exact7_source_closure", {})
    if len(exact7.get("records", [])) != 7 or csha(exact7["records"]) != exact7.get("canonical_records_digest_sha256") or exact7["canonical_records_digest_sha256"] != contract["exact7_records_digest_sha256"]:
        raise RuntimeError("exact7 digest")
    for value in exact7["records"]:
        record({key: value[key] for key in ("path", "sha256", "logical_bytes")})

    authority_root = args.authority_root
    if authority_root != authority_root.resolve() or str(authority_root) != contract["lineage"]["authority_root"]:
        raise RuntimeError("fresh authority root")
    prep = authority_root.with_name(authority_root.name + ".registration-prep")
    if os.path.lexists(authority_root) or os.path.lexists(prep):
        raise FileExistsError("authority state")
    for value in contract["historical_pre_materialization_absences"].values():
        if os.path.lexists(value["path"]):
            raise RuntimeError(f"historical absence: {value['path']}")
    if live_relevant_processes():
        raise RuntimeError("live reconciliation")

    files = {
        "authority_contract": str(args.contract),
        "authority_materializer": str(args.materializer_source),
        "formal": formal_record["path"],
        "deployed_transcript": transcript["path"],
        "failure_receipt": failure_receipt_record["path"],
    }
    for name, value in source_records.items():
        files[f"source::{name}"] = value["path"]
    for name, value in superseded_source_records.items():
        files[f"superseded_source::{name}"] = value["path"]
    for rel, _, _ in failure_tree["inventory"]:
        files[f"failure_evidence::{rel}"] = str(Path(contract["superseded_authority_materialization"]["evidence_tree"]["root"]) / rel)
    for index, value in enumerate(exact7["records"]):
        files[f"exact7::{index}::{value['role']}"] = value["path"]
    trees = {
        "f813_REG": contract["reconciliation_registration_tree"]["root"],
        "old_b73_REG": contract["old_b73_registration_tree"]["root"],
        "failed_authority_materialization_evidence": contract["superseded_authority_materialization"]["evidence_tree"]["root"],
    }
    absences = {f"historical::{name}": value["path"] for name, value in sorted(contract["historical_pre_materialization_absences"].items())}
    absences["authority_prep"] = str(prep)
    pre = snapshot(files, trees, absences)
    post = snapshot(files, trees, absences)
    if pre != post:
        raise RuntimeError("input drift")
    stable_absences = {key: value for key, value in absences.items() if Path(value) != prep}
    stable_pre = snapshot(files, trees, stable_absences)

    checks = {name: True for name in contract["authority_receipt_contract"]["check_keys"]}
    receipt = {
        "format": OUTPUT_FORMAT,
        "status": OUTPUT_STATUS,
        "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": materializer_record,
        "future_execution_wrapper_source": source_records["execution_wrapper"],
        "reconciliation_preregistration": formal_record,
        "reconciliation_design_contract": source_records["reconciliation_design_contract"],
        "reconciliation_registration_materializer": source_records["reconciliation_registration_materializer"],
        "reconciler_source": source_records["reconciler"],
        "superseded_authority_materialization": {**contract["superseded_authority_materialization"], "observed_evidence_tree": failure_tree},
        "transcript_deployment": {"record": transcript, "missing_during_superseded_attempt": True, "present_before_fresh_authority_materialization": True},
        "old_b73_tree": old_tree,
        "reconciliation_REG_exact2": reg_tree,
        "exact7_source_closure": {"records": exact7["records"], "records_digest_sha256": exact7["canonical_records_digest_sha256"]},
        "attempt_root": contract["lineage"]["attempt_root"],
        "transparent_static_receipt_path": contract["lineage"]["transparent_static_receipt_path"],
        "historical_pre_materialization_absences": {name: {**value, "absent": True} for name, value in sorted(contract["historical_pre_materialization_absences"].items())},
        "required_absences": {name: {**value, "absent": True} for name, value in sorted(contract["required_current_absences_after_authority"].items())},
        "checks": checks,
        "checks_sha256": csha(checks),
        "input_pre_snapshot": pre,
        "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True,
        "authorization": contract["authority_receipt_contract"]["authorization_exact"],
        "runtime_observation": contract["authority_receipt_contract"]["runtime_observation_exact"],
    }
    if set(receipt) != set(contract["authority_receipt_contract"]["required_top_level_keys"]):
        raise RuntimeError("receipt keyset")
    if csha(sorted(checks)) != contract["authority_receipt_contract"]["check_key_set_sha256"] or csha(checks) != contract["authority_receipt_contract"]["checks_sha256"]:
        raise RuntimeError("receipt checks")

    atomic_promotion_fixture()
    prep.mkdir()
    try:
        atomic_json(prep / "authority_receipt.json", receipt)
        fsync_dir(prep)
        if snapshot(files, trees, stable_absences) != stable_pre:
            raise RuntimeError("pre-promote drift")
        os.replace(prep, authority_root)
        fsync_dir(authority_root.parent)
    except BaseException:
        if prep.exists() and not authority_root.exists():
            shutil.rmtree(prep)
        raise
    output = authority_root / "authority_receipt.json"
    print(json.dumps({"path": str(output), "sha256": sha(output), "logical_bytes": output.stat().st_size, "reconciliation_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
