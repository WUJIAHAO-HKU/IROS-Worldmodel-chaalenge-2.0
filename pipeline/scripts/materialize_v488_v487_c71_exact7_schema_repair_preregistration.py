#!/usr/bin/env python3
"""Register a nonauthorizing v488 c71 exact7-schema repair formal; never run r2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CONTRACT_PATH = ROOT / "pipeline/scripts/v488_v487_c71_exact7_schema_repair_contract.json"
CONTRACT_FORMAT = "strict-track2-v488-v487-c71-exact7-schema-repair-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
FORMAL_FORMAT = "strict-track2-v488-v487-c71-exact7-schema-repair-preregistration-v1"
FORMAL_STATUS = "preregistered_readonly_source_repair_pending_postregistration_authority"


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
        raise RuntimeError(f"canonical regular file: {path}")
    if sha(path) != digest or path.stat().st_size != size:
        raise RuntimeError(f"file closure: {path}")
    return {"path": str(path), "sha256": digest, "logical_bytes": size}


def record(value: dict) -> dict:
    if set(value) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record schema")
    return regular(value["path"], value["sha256"], value["logical_bytes"])


def exact_tree(root: Path | str) -> dict:
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


def verify_tree(spec: dict) -> dict:
    observed = exact_tree(spec["root"])
    expected = {key: spec[key] for key in observed}
    if observed != expected:
        raise RuntimeError(f"tree mismatch: {spec['root']}")
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
    for name, text in sorted(files.items()):
        path = Path(text)
        if path != path.resolve() or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"snapshot file: {name}")
        file_rows.append({"name": name, "path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size})
    tree_rows = {name: exact_tree(path) for name, path in sorted(trees.items())}
    absence_rows = []
    for name, text in sorted(absences.items()):
        if os.path.lexists(text):
            raise RuntimeError(f"snapshot absence: {name}")
        absence_rows.append({"name": name, "path": str(Path(text)), "absent": True})
    body = {"files": file_rows, "trees": tree_rows, "absences": absence_rows}
    return {**body, "snapshot_sha256": csha(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--registration-root", type=Path, required=True)
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract canonical path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS or pending(contract):
        raise RuntimeError("contract frozen schema")
    materializer_record = regular(args.materializer_source, args.materializer_sha, args.materializer_source.stat().st_size)
    if args.materializer_source.resolve() != Path(__file__).resolve() or materializer_record != contract["materializer_source"]:
        raise RuntimeError("materializer source binding")
    if contract.get("current_design_authorization") != {
        "repair_formal_materialization_authorized": False,
        "readonly_reconciliation_authorized": False,
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
        raise RuntimeError("design authority")

    sources = {name: record(value) for name, value in contract["source_closure"].items()}
    parent = json.loads(Path(sources["parent_v485_preregistration"]["path"]).read_text())
    f813 = json.loads(Path(sources["f813_reconciliation_preregistration"]["path"]).read_text())
    f813_tree = verify_tree(contract["f813_registration_tree"])
    authority_tree = verify_tree(contract["v487_authority_tree"])
    helper_tree = verify_tree(contract["v487_helper_evidence_tree"])
    failed_tree = verify_tree(contract["failed_reconciliation_attempt_tree"])
    terminal = json.loads(Path(contract["failed_reconciliation_attempt_tree"]["root"], "terminal_receipt.json").read_text())
    if terminal.get("format") != "strict-track2-v487-v486-v485-static-reconciliation-attempt-terminal-v1" or terminal.get("status") != "failed_no_retry" or terminal.get("passed") is not False or terminal.get("retry_authorized") is not False:
        raise RuntimeError("failed attempt terminal")
    stderr = Path(contract["failed_reconciliation_attempt_tree"]["root"], "reconciler_stderr.log").read_text(errors="replace")
    if "reconciliation exact7 closure" not in stderr or "line 300" not in stderr:
        raise RuntimeError("failed attempt root cause")

    roles = contract["exact7_schema_repair_proof"]["roles_in_order"]
    expected_records = [{"role": role, **parent["execution_sources"][role]} for role in roles]
    observed_records = []
    for expected in expected_records:
        observed = record({key: expected[key] for key in ("path", "sha256", "logical_bytes")})
        observed_record = {"role": expected["role"], **observed}
        if observed_record != expected:
            raise RuntimeError(f"exact7 current source: {expected['role']}")
        observed_records.append(observed_record)
    expected_map = {row["role"]: {key: row[key] for key in ("path", "sha256", "logical_bytes")} for row in expected_records}
    expected_digest = csha(expected_records)
    actual = f813.get("exact7_source_closure")
    if set(actual or {}) != {"all_records_must_equal_parent_formal_and_old_receipt_and_current_files", "canonical_records_digest_sha256", "records", "roles_in_order"}:
        raise RuntimeError("f813 exact4 schema")
    if actual["all_records_must_equal_parent_formal_and_old_receipt_and_current_files"] is not True or actual["roles_in_order"] != roles or actual["records"] != expected_records or observed_records != expected_records or actual["canonical_records_digest_sha256"] != expected_digest or expected_map != parent["execution_sources"]:
        raise RuntimeError("f813 exact7 semantic closure")
    if expected_digest != contract["exact7_schema_repair_proof"]["canonical_records_digest_sha256"]:
        raise RuntimeError("exact7 frozen digest")

    root = args.registration_root
    if root != root.resolve() or str(root) != contract["lineage"]["registration_root"] or not root.parent.is_dir() or root.parent.is_symlink():
        raise RuntimeError("registration root")
    prep = root.with_name(root.name + ".registration-prep")
    if os.path.lexists(root) or os.path.lexists(prep):
        raise FileExistsError("registration state")
    for value in contract["required_absences_before_registration"].values():
        if os.path.lexists(value["path"]):
            raise RuntimeError(f"required absence: {value['path']}")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source)}
    for name, value in sources.items():
        files[f"source::{name}"] = value["path"]
    for value in observed_records:
        files[f"exact7_current::{value['role']}"] = value["path"]
    for rel, _, _ in failed_tree["inventory"]:
        files[f"failed_attempt::{rel}"] = str(Path(contract["failed_reconciliation_attempt_tree"]["root"]) / rel)
    trees = {
        "f813_REG": contract["f813_registration_tree"]["root"],
        "v487_authority_REG": contract["v487_authority_tree"]["root"],
        "v487_helper_evidence": contract["v487_helper_evidence_tree"]["root"],
        "failed_reconciliation_attempt": contract["failed_reconciliation_attempt_tree"]["root"],
    }
    absences = {name: value["path"] for name, value in contract["required_absences_before_registration"].items()}
    absences["registration_prep"] = str(prep)
    pre = snapshot(files, trees, absences)
    post = snapshot(files, trees, absences)
    if pre != post:
        raise RuntimeError("input drift")
    stable_absences = {name: path for name, path in absences.items() if Path(path) != prep}
    stable_pre = snapshot(files, trees, stable_absences)

    formal = {
        "format": FORMAL_FORMAT,
        "status": FORMAL_STATUS,
        "seed": contract["lineage"]["seed"],
        "design_contract": contract_record,
        "materializer_source": materializer_record,
        "source_closure": sources,
        "source_closure_sha256": csha(sources),
        "f813_registration_tree": f813_tree,
        "v487_authority_tree": authority_tree,
        "v487_helper_evidence_tree": helper_tree,
        "failed_reconciliation_attempt_tree": failed_tree,
        "failed_attempt_no_retry": True,
        "transparent_static_receipt_path": contract["lineage"]["transparent_static_receipt_path"],
        "exact7_schema_repair_proof": {
            "actual_exact4_keyset": sorted(actual),
            "old_c71_expected_exact3_keyset": ["execution_source_records", "execution_sources", "execution_sources_digest_sha256"],
            "records_equal": True,
            "observed_current_records": observed_records,
            "observed_current_records_digest_sha256": csha(observed_records),
            "roles_equal": True,
            "digest_equal": True,
            "map_equal": True,
            "canonical_records_digest_sha256": expected_digest,
            "new_r2_required_rule": contract["exact7_schema_repair_proof"]["new_r2_required_rule"],
        },
        "future_reconciler_r2_source": sources["future_reconciler_r2"],
        "required_postregistration_authority": contract["lineage"]["postregistration_authority_required"],
        "authorization": contract["formal_authorization_exact"],
        "runtime_observation": contract["runtime_observation_exact"],
        "input_pre_snapshot": pre,
        "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True,
    }
    if pending(formal):
        raise RuntimeError("formal placeholder")

    prep.mkdir()
    try:
        atomic_json(prep / "preregistration.json", formal)
        fsync_dir(prep)
        if snapshot(files, trees, stable_absences) != stable_pre:
            raise RuntimeError("pre-promote input drift")
        os.replace(prep, root)
        fsync_dir(root.parent)
    except BaseException:
        if prep.exists() and not root.exists():
            shutil.rmtree(prep)
        raise
    output = root / "preregistration.json"
    print(json.dumps({"path": str(output), "sha256": sha(output), "logical_bytes": output.stat().st_size, "authorization": formal["authorization"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
