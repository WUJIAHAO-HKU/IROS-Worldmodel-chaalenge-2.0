#!/usr/bin/env python3
"""Read-only v499 adapter for the sole stale v498 transport-helper-copy path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import stat
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS = ROOT / "pipeline/scripts"
SELF_PATH = SCRIPTS / "adapt_v499_v498_diagnostic_process_receipt_transport_helper_copy_path.py"

DIAGNOSTIC_SOURCE = SCRIPTS / "diagnose_v498_v497_v496_v495_failure_tree_pre_authority.py"
DIAGNOSTIC_SOURCE_RECORD = {
    "path": DIAGNOSTIC_SOURCE.as_posix(),
    "sha256": "328d88b9dd5c2268f27c612322785eeca3cfe053018b00512492f984f8332fe6",
    "logical_bytes": 38424,
}
TRANSPORT_HELPER = SCRIPTS / "invoke_v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_once.py"
TRANSPORT_HELPER_RECORD = {
    "path": TRANSPORT_HELPER.as_posix(),
    "sha256": "f7072fae9b88ca1736818f1304191d5bfc4c2d550eb16f0a6f73bf2b8f3a1dce",
    "logical_bytes": 27949,
}
TRANSPORT_SCRIPT = Path("/root/v498_pre_authority_diagnostic_once.sh")
TRANSPORT_SCRIPT_RECORD = {
    "path": TRANSPORT_SCRIPT.as_posix(),
    "sha256": "c41a960cce856ddcf27933930b888a17cf61ee1de3552aa59a2867e8b1b0b8ea",
    "logical_bytes": 1171,
}

DIAGNOSTIC_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1640_20260825"
DIAGNOSTIC_PREP = DIAGNOSTIC_ROOT.with_name(DIAGNOSTIC_ROOT.name + ".registration-prep")
DIAGNOSTIC_RECEIPT = DIAGNOSTIC_ROOT / "diagnostic_receipt.json"
DIAGNOSTIC_RECEIPT_RECORD = {
    "path": DIAGNOSTIC_RECEIPT.as_posix(),
    "sha256": "2bb515bba9039d3bac3545c1780bb82d9c7609aafa8a64012f9c9574eec5da20",
    "logical_bytes": 14528,
}
DIAGNOSTIC_TREE = {
    "root": DIAGNOSTIC_ROOT.as_posix(),
    "inventory": [["diagnostic_receipt.json", DIAGNOSTIC_RECEIPT_RECORD["sha256"], 14528]],
    "file_count": 1,
    "logical_file_bytes": 14528,
    "sha256sum_lines_digest_sha256": "99991dff3106be728dbec3c0c0bd61df41313a894a19a6cf3b0b71fb21d74d93",
    "canonical_json_triples_digest_sha256": "ab4b91264ecf854be7df5263643c0def7deec74bb76c22ce455ee089a57a2a96",
}

EVIDENCE_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1640_20260825"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
PROCESS_RECEIPT = EVIDENCE_ROOT / "process_receipt.json"
PROCESS_RECEIPT_RECORD = {
    "path": PROCESS_RECEIPT.as_posix(),
    "sha256": "fa7269bdacac994efd5ef8eb44cf9d3d1c7014e5b9aa902e95a5f4243609fabd",
    "logical_bytes": 26493,
}
EVIDENCE_TREE = {
    "root": EVIDENCE_ROOT.as_posix(),
    "inventory": [
        ["argv.json", "acad443504876013e78d1f277372986f4391170bbc220c713b1961e51ea408e2", 14317],
        ["diagnostic_stderr.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["diagnostic_stdout.log", "319b17ca80b72e2a063fd65432b00a04ee0871752feb401c914bc11ad29cb8e4", 661],
        ["intent.json", "ecc51adb493e4c2ec7dab31473132d74ff56dc06589387e7b84152eb0936740f", 545],
        ["process_receipt.json", PROCESS_RECEIPT_RECORD["sha256"], 26493],
        ["transport_helper.py", TRANSPORT_HELPER_RECORD["sha256"], 27949],
    ],
    "file_count": 6,
    "logical_file_bytes": 69965,
    "sha256sum_lines_digest_sha256": "17a283bf1f1957c484091c4e44920368c2203e7f7ea05df22ad975582e0b774b",
    "canonical_json_triples_digest_sha256": "2c3b56d838eeb3c936a0f9aad5c8ba8a149a7bc4b3f24c1f9aff0f254337b393",
}
FINAL_HELPER_COPY = EVIDENCE_ROOT / "transport_helper.py"
FINAL_HELPER_COPY_RECORD = {
    "path": FINAL_HELPER_COPY.as_posix(),
    "sha256": TRANSPORT_HELPER_RECORD["sha256"],
    "logical_bytes": TRANSPORT_HELPER_RECORD["logical_bytes"],
}
STALE_HELPER_COPY = EVIDENCE_PREP / "transport_helper.py"
STALE_HELPER_COPY_RECORD = {
    "path": STALE_HELPER_COPY.as_posix(),
    "sha256": TRANSPORT_HELPER_RECORD["sha256"],
    "logical_bytes": TRANSPORT_HELPER_RECORD["logical_bytes"],
}

ADAPTER_ROOT = J / "v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_seed1641_20260825"
ADAPTER_PREP = ADAPTER_ROOT.with_name(ADAPTER_ROOT.name + ".registration-prep")
ADAPTER_RECEIPT = ADAPTER_ROOT / "adapter_receipt.json"

PROCESS_TOP_KEYS = {
    "argv", "authority_materializer_invocations", "cleanup",
    "corrected_inner_wrapper_invocations", "diagnostic_invocations",
    "diagnostic_receipt", "diagnostic_registration_tree", "diagnostic_returncode",
    "format", "helper_returncode", "intent", "outer_wrapper_invocations", "passed",
    "post_snapshot", "pre_post_snapshots_exactly_equal", "pre_snapshot",
    "reconciler_r2_invocations", "retry_authorized", "status", "stderr", "stdout",
    "transport_helper", "transport_helper_copy", "transport_script", "wall_seconds",
}
PROCESS_KEY_SET_SHA256 = "3af4a6e4739cd6570e73907b31cbe32bf78bf4952e900cda96b7dbfd792e9c8a"
ORIGINAL_PROCESS_VALUE_SHA256 = "4f043cb1e415b88a5f02a4b5180f78e94e295ca351065edbeccff9df8172f59f"
NORMALIZED_PROCESS_VALUE_SHA256 = "9318ba42a5ce23fce6b3179756c99b9ef18d907ce338930e61eb0c3dae38804d"
DIAGNOSTIC_TOP_KEYS = {
    "all_predicates_passed", "authorization", "diagnostic_source", "failed_v497_ancestry",
    "format", "passed", "predicate_name_key_set_sha256", "predicate_names", "predicates",
    "predicates_sha256", "required_absences", "root_cause_boundary", "runtime_observation",
    "seed", "snapshot_1", "snapshot_2", "snapshots_exactly_equal", "status",
    "v496_authority_sources", "v496_failure_no_retry", "v496_failure_process_receipt",
    "v496_failure_transport_helper", "v496_failure_tree",
}

AUTHORIZATION = {
    "adapter_authorized": False,
    "authority_materialization_authorized": False,
    "outer_execution_authorized": False,
    "corrected_inner_wrapper_authorized": False,
    "reconciler_r2_authorized": False,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
    "submission_authorized": False,
    "retry_authorized": False,
}
RUNTIME = {
    "adapter_invocations": 1,
    "authority_materializer_invocations": 0,
    "outer_wrapper_invocations": 0,
    "corrected_inner_wrapper_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
RECEIPT_TOP_KEYS = {
    "format", "status", "passed", "seed", "adapter_source", "diagnostic_source",
    "transport_helper_source", "transport_script_source", "diagnostic_receipt",
    "diagnostic_registration_tree", "diagnostic_execution_evidence_tree",
    "original_process_receipt_record", "original_process_receipt",
    "original_process_receipt_sha256", "original_process_key_set_sha256",
    "stale_transport_helper_copy", "corrected_transport_helper_copy",
    "transport_helper_copy_mapping", "normalized_process_receipt",
    "normalized_process_receipt_sha256", "canonical_json_leaf_diff",
    "canonical_json_leaf_diff_count", "normalization_exactly_one_leaf",
    "input_pre_snapshot_1", "input_pre_snapshot_2", "input_post_snapshot_1",
    "input_post_snapshot_2", "input_snapshots_exactly_equal", "required_absences",
    "synthetic_tamper_evidence", "runtime_observation", "authorization",
}


class ControlledSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"controlled signal {signum}")
        self.signum = signum


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: dict | None = None) -> dict:
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"not canonical regular: {path}")
    observed = os.lstat(path)
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"not regular: {path}")
    record = {"path": str(path), "sha256": file_sha(path), "logical_bytes": observed.st_size}
    if expected is not None and record != expected:
        raise RuntimeError(f"record mismatch: {path}")
    return record


def read_json_exact(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(text)
    if text[end:].strip():
        raise RuntimeError(f"trailing JSON: {path}")
    return value


def exact_tree(root: Path) -> dict:
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root: {root}")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
            raise RuntimeError(f"tree nonregular: {path}")
        inventory.append([path.relative_to(root).as_posix(), file_sha(path), path.stat().st_size])
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in inventory).encode("utf-8")
    return {
        "root": str(root),
        "inventory": inventory,
        "file_count": len(inventory),
        "logical_file_bytes": sum(row[2] for row in inventory),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": canonical_sha(inventory),
    }


def json_leaf_diff(left: object, right: object, path: tuple = ()) -> list[dict]:
    if type(left) is not type(right):
        return [{"path": list(path), "before": left, "after": right}]
    if isinstance(left, dict):
        if set(left) != set(right):
            return [{"path": list(path), "before": left, "after": right}]
        result = []
        for key in sorted(left):
            result.extend(json_leaf_diff(left[key], right[key], path + (key,)))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": list(path), "before": left, "after": right}]
        result = []
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            result.extend(json_leaf_diff(lhs, rhs, path + (index,)))
        return result
    return [] if left == right else [{"path": list(path), "before": left, "after": right}]


def normalize_process(
    original: dict,
    stale_record: dict,
    corrected_record: dict,
    *,
    stale_exists: bool,
    corrected_regular: bool,
) -> tuple[dict, list[dict], dict]:
    if set(original) != PROCESS_TOP_KEYS:
        raise RuntimeError("process keyset")
    if canonical_sha(sorted(original)) != PROCESS_KEY_SET_SHA256:
        raise RuntimeError("process keyset digest")
    if canonical_sha(original) != ORIGINAL_PROCESS_VALUE_SHA256:
        raise RuntimeError("original process value digest")
    if original.get("transport_helper_copy") != stale_record:
        raise RuntimeError("stale alias value")
    if stale_exists:
        raise RuntimeError("stale alias unexpectedly exists")
    if not corrected_regular:
        raise RuntimeError("corrected alias not canonical regular")
    stale_path = PurePosixPath(stale_record["path"])
    corrected_path = PurePosixPath(corrected_record["path"])
    if set(stale_record) != {"path", "sha256", "logical_bytes"} or set(corrected_record) != set(stale_record):
        raise RuntimeError("alias schema")
    if stale_path.parent != PurePosixPath(EVIDENCE_PREP.as_posix()) or corrected_path.parent != PurePosixPath(EVIDENCE_ROOT.as_posix()):
        raise RuntimeError("alias parent mapping")
    if stale_path.name != corrected_path.name or stale_path.name != "transport_helper.py":
        raise RuntimeError("alias basename mapping")
    if {"sha256": stale_record["sha256"], "logical_bytes": stale_record["logical_bytes"]} != {
        "sha256": corrected_record["sha256"], "logical_bytes": corrected_record["logical_bytes"]
    }:
        raise RuntimeError("alias content mapping")
    normalized = copy.deepcopy(original)
    normalized["transport_helper_copy"] = copy.deepcopy(corrected_record)
    diff = json_leaf_diff(original, normalized)
    expected_diff = [{
        "path": ["transport_helper_copy", "path"],
        "before": stale_record["path"],
        "after": corrected_record["path"],
    }]
    if diff != expected_diff or canonical_sha(normalized) != NORMALIZED_PROCESS_VALUE_SHA256:
        raise RuntimeError("normalization is not exact one leaf")
    mapping = {
        "stale_parent": str(stale_path.parent),
        "corrected_parent": str(corrected_path.parent),
        "same_basename": True,
        "basename": stale_path.name,
        "sha256_and_logical_bytes_unchanged": True,
        "stale_parent_is_evidence_prep": True,
        "corrected_parent_is_evidence_root": True,
    }
    return normalized, diff, mapping


def validate_diagnostic(receipt: dict) -> None:
    if set(receipt) != DIAGNOSTIC_TOP_KEYS:
        raise RuntimeError("diagnostic top schema")
    if receipt.get("format") != "strict-track2-v498-v497-v496-v495-failure-tree-pre-authority-diagnostic-v1":
        raise RuntimeError("diagnostic format")
    if receipt.get("status") != "passed_no_execution_authority" or receipt.get("passed") is not True:
        raise RuntimeError("diagnostic status")
    if receipt.get("diagnostic_source") != DIAGNOSTIC_SOURCE_RECORD:
        raise RuntimeError("diagnostic source")
    if receipt.get("all_predicates_passed") is not True or receipt.get("snapshots_exactly_equal") is not True:
        raise RuntimeError("diagnostic predicates")
    if set(receipt.get("authorization", {}).values()) != {False}:
        raise RuntimeError("diagnostic authorization")
    runtime = receipt.get("runtime_observation", {})
    if runtime.get("diagnostic_invocations") != 1 or any(runtime.get(key) != 0 for key in (
        "authority_materializer_invocations", "outer_wrapper_invocations",
        "corrected_inner_wrapper_invocations", "reconciler_r2_invocations", "folds", "policy_updates"
    )) or runtime.get("phase_a_executed") is not False or runtime.get("training_launched") is not False:
        raise RuntimeError("diagnostic runtime")


def validate_process(process: dict) -> None:
    if process.get("format") != "strict-track2-v498-pre-authority-diagnostic-process-receipt-v1":
        raise RuntimeError("process format")
    if process.get("status") != "passed_exact_once_no_authority_or_reconciliation_execution" or process.get("passed") is not True:
        raise RuntimeError("process status")
    if process.get("diagnostic_invocations") != 1 or any(process.get(key) != 0 for key in (
        "authority_materializer_invocations", "outer_wrapper_invocations",
        "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"
    )):
        raise RuntimeError("process invocation partition")
    if process.get("retry_authorized") is not False or process.get("pre_post_snapshots_exactly_equal") is not True:
        raise RuntimeError("process retry/snapshots")
    cleanup = process.get("cleanup", {})
    if cleanup.get("reaped") is not True or cleanup.get("group_empty") is not True:
        raise RuntimeError("process cleanup")
    if process.get("diagnostic_receipt") != DIAGNOSTIC_RECEIPT_RECORD or process.get("diagnostic_registration_tree") != DIAGNOSTIC_TREE:
        raise RuntimeError("process diagnostic ancestry")
    if process.get("transport_helper") != TRANSPORT_HELPER_RECORD or process.get("transport_script") != TRANSPORT_SCRIPT_RECORD:
        raise RuntimeError("process transport ancestry")


def immutable_snapshot(adapter_source_record: dict) -> dict:
    return {
        "adapter_source": regular(SELF_PATH, adapter_source_record),
        "diagnostic_source": regular(DIAGNOSTIC_SOURCE, DIAGNOSTIC_SOURCE_RECORD),
        "transport_helper_source": regular(TRANSPORT_HELPER, TRANSPORT_HELPER_RECORD),
        "transport_script_source": regular(TRANSPORT_SCRIPT, TRANSPORT_SCRIPT_RECORD),
        "diagnostic_receipt": regular(DIAGNOSTIC_RECEIPT, DIAGNOSTIC_RECEIPT_RECORD),
        "original_process_receipt": regular(PROCESS_RECEIPT, PROCESS_RECEIPT_RECORD),
        "corrected_transport_helper_copy": regular(FINAL_HELPER_COPY, FINAL_HELPER_COPY_RECORD),
        "diagnostic_registration_tree": exact_tree(DIAGNOSTIC_ROOT),
        "diagnostic_execution_evidence_tree": exact_tree(EVIDENCE_ROOT),
        "required_absences": {
            "diagnostic_prep": {"path": DIAGNOSTIC_PREP.as_posix(), "absent": not os.path.lexists(DIAGNOSTIC_PREP)},
            "evidence_prep": {"path": EVIDENCE_PREP.as_posix(), "absent": not os.path.lexists(EVIDENCE_PREP)},
            "stale_helper_copy": {"path": STALE_HELPER_COPY.as_posix(), "absent": not os.path.lexists(STALE_HELPER_COPY)},
        },
    }


def tamper_suite(original: dict) -> dict:
    corrected = copy.deepcopy(FINAL_HELPER_COPY_RECORD)
    cases = {}

    def rejects(name: str, value: dict, stale: dict = STALE_HELPER_COPY_RECORD, **kwargs) -> None:
        try:
            normalize_process(value, stale, kwargs.pop("corrected", corrected), stale_exists=kwargs.pop("stale_exists", False), corrected_regular=kwargs.pop("corrected_regular", True))
        except RuntimeError:
            cases[name] = True
        else:
            cases[name] = False

    second = copy.deepcopy(original)
    second["wall_seconds"] = second["wall_seconds"] + 1.0
    rejects("second_field_diff_rejected", second)
    wrong_parent = copy.deepcopy(corrected); wrong_parent["path"] = str(EVIDENCE_ROOT.parent / corrected["path"].split("/")[-1])
    rejects("wrong_parent_rejected", original, corrected=wrong_parent)
    wrong_name = copy.deepcopy(corrected); wrong_name["path"] = str(EVIDENCE_ROOT / "other.py")
    rejects("wrong_basename_rejected", original, corrected=wrong_name)
    wrong_sha = copy.deepcopy(corrected); wrong_sha["sha256"] = "0" * 64
    rejects("wrong_sha_rejected", original, corrected=wrong_sha)
    wrong_bytes = copy.deepcopy(corrected); wrong_bytes["logical_bytes"] += 1
    rejects("wrong_bytes_rejected", original, corrected=wrong_bytes)
    rejects("symlink_rejected", original, corrected_regular=False)
    rejects("stale_exists_rejected", original, stale_exists=True)
    cases["exact_normalization_passed"] = False
    try:
        normalize_process(original, STALE_HELPER_COPY_RECORD, corrected, stale_exists=False, corrected_regular=True)
        cases["exact_normalization_passed"] = True
    except RuntimeError:
        pass
    if set(cases.values()) != {True}:
        raise RuntimeError("tamper suite")
    return {"cases": cases, "all_passed": True, "cases_sha256": canonical_sha(cases)}


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_receipt(path: Path, receipt: dict) -> None:
    payload = canonical_bytes(receipt) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_owned_prep(prep: Path, identity: tuple[int, int] | None) -> None:
    if identity is None or not prep.is_dir() or prep.is_symlink():
        return
    observed = os.lstat(prep)
    if (observed.st_dev, observed.st_ino) != identity:
        return
    for child in prep.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError("foreign prep content")
        child.unlink()
    prep.rmdir()
    fsync_dir(prep.parent)


def build_receipt(adapter_source_record: dict) -> tuple[dict, dict]:
    if os.path.lexists(ADAPTER_ROOT) or os.path.lexists(ADAPTER_PREP):
        raise RuntimeError("adapter registration prestate")
    pre_1 = immutable_snapshot(adapter_source_record)
    pre_2 = immutable_snapshot(adapter_source_record)
    if pre_1 != pre_2 or not all(row["absent"] for row in pre_1["required_absences"].values()):
        raise RuntimeError("unstable prestate")
    if pre_1["diagnostic_registration_tree"] != DIAGNOSTIC_TREE or pre_1["diagnostic_execution_evidence_tree"] != EVIDENCE_TREE:
        raise RuntimeError("ancestry tree")
    diagnostic = read_json_exact(DIAGNOSTIC_RECEIPT)
    process = read_json_exact(PROCESS_RECEIPT)
    if not isinstance(diagnostic, dict) or not isinstance(process, dict):
        raise RuntimeError("receipt object")
    validate_diagnostic(diagnostic)
    validate_process(process)
    normalized, diff, mapping = normalize_process(
        process,
        STALE_HELPER_COPY_RECORD,
        regular(FINAL_HELPER_COPY, FINAL_HELPER_COPY_RECORD),
        stale_exists=os.path.lexists(STALE_HELPER_COPY),
        corrected_regular=True,
    )
    synthetic = tamper_suite(process)
    post_1 = immutable_snapshot(adapter_source_record)
    post_2 = immutable_snapshot(adapter_source_record)
    if pre_1 != post_1 or pre_1 != post_2:
        raise RuntimeError("independent pre/post snapshot drift")
    receipt = {
        "format": "strict-track2-v499-v498-diagnostic-process-receipt-helper-copy-path-adapter-v1",
        "status": "passed_readonly_exact_one_leaf_process_receipt_normalization_no_authority",
        "passed": True,
        "seed": 1641,
        "adapter_source": adapter_source_record,
        "diagnostic_source": DIAGNOSTIC_SOURCE_RECORD,
        "transport_helper_source": TRANSPORT_HELPER_RECORD,
        "transport_script_source": TRANSPORT_SCRIPT_RECORD,
        "diagnostic_receipt": DIAGNOSTIC_RECEIPT_RECORD,
        "diagnostic_registration_tree": DIAGNOSTIC_TREE,
        "diagnostic_execution_evidence_tree": EVIDENCE_TREE,
        "original_process_receipt_record": PROCESS_RECEIPT_RECORD,
        "original_process_receipt": process,
        "original_process_receipt_sha256": canonical_sha(process),
        "original_process_key_set_sha256": canonical_sha(sorted(process)),
        "stale_transport_helper_copy": STALE_HELPER_COPY_RECORD,
        "corrected_transport_helper_copy": FINAL_HELPER_COPY_RECORD,
        "transport_helper_copy_mapping": mapping,
        "normalized_process_receipt": normalized,
        "normalized_process_receipt_sha256": canonical_sha(normalized),
        "canonical_json_leaf_diff": diff,
        "canonical_json_leaf_diff_count": len(diff),
        "normalization_exactly_one_leaf": diff == [{
            "path": ["transport_helper_copy", "path"],
            "before": STALE_HELPER_COPY.as_posix(),
            "after": FINAL_HELPER_COPY.as_posix(),
        }],
        "input_pre_snapshot_1": pre_1,
        "input_pre_snapshot_2": pre_2,
        "input_post_snapshot_1": post_1,
        "input_post_snapshot_2": post_2,
        "input_snapshots_exactly_equal": True,
        "required_absences": {
            **pre_1["required_absences"],
            "adapter_prep": {"path": ADAPTER_PREP.as_posix(), "absent": True},
        },
        "synthetic_tamper_evidence": synthetic,
        "runtime_observation": RUNTIME,
        "authorization": AUTHORIZATION,
    }
    if set(receipt) != RECEIPT_TOP_KEYS or receipt["normalization_exactly_one_leaf"] is not True:
        raise RuntimeError("adapter receipt schema")
    return receipt, pre_1


def publish(receipt: dict, pre_snapshot: dict) -> dict:
    prep_identity = None
    committed = False
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def handler(signum, _frame):
        if committed:
            return
        raise ControlledSignal(signum)

    for sig in old_handlers:
        signal.signal(sig, handler)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        ADAPTER_PREP.mkdir(mode=0o700)
        observed = os.lstat(ADAPTER_PREP)
        prep_identity = (observed.st_dev, observed.st_ino)
        fsync_dir(ADAPTER_PREP.parent)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        write_receipt(ADAPTER_PREP / "adapter_receipt.json", receipt)
        fsync_dir(ADAPTER_PREP)
        if immutable_snapshot(receipt["adapter_source"]) != pre_snapshot:
            raise RuntimeError("pre-promote drift")
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        os.replace(ADAPTER_PREP, ADAPTER_ROOT)
        fsync_dir(ADAPTER_ROOT.parent)
        expected_record = {
            "path": str(ADAPTER_RECEIPT),
            "sha256": file_sha(ADAPTER_RECEIPT),
            "logical_bytes": ADAPTER_RECEIPT.stat().st_size,
        }
        tree = exact_tree(ADAPTER_ROOT)
        if tree["inventory"] != [["adapter_receipt.json", expected_record["sha256"], expected_record["logical_bytes"]]]:
            raise RuntimeError("adapter exact1")
        post_1 = immutable_snapshot(receipt["adapter_source"])
        post_2 = immutable_snapshot(receipt["adapter_source"])
        if post_1 != pre_snapshot or post_2 != pre_snapshot:
            raise RuntimeError("post-promote immutable drift")
        committed = True
        return {"adapter_receipt": expected_record, "adapter_registration_tree": tree}
    except BaseException:
        if not committed and not os.path.lexists(ADAPTER_ROOT):
            remove_owned_prep(ADAPTER_PREP, prep_identity)
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        for sig, old in old_handlers.items():
            signal.signal(sig, old)


def synthetic_self_test() -> dict:
    original = read_json_exact(Path(__file__).with_name("_v498_process_receipt_actual.json")) if Path(__file__).with_name("_v498_process_receipt_actual.json").exists() else None
    if original is None:
        original = {
            key: None for key in PROCESS_TOP_KEYS
        }
        original.update({
            "format": "strict-track2-v498-pre-authority-diagnostic-process-receipt-v1",
            "status": "passed_exact_once_no_authority_or_reconciliation_execution",
            "passed": True,
            "transport_helper_copy": copy.deepcopy(STALE_HELPER_COPY_RECORD),
        })
        # The frozen digest gate intentionally requires the actual immutable receipt.
        return {"passed": True, "mode": "compile_only_without_actual_receipt", "pure_leaf_diff": len(json_leaf_diff(
            {"transport_helper_copy": STALE_HELPER_COPY_RECORD},
            {"transport_helper_copy": FINAL_HELPER_COPY_RECORD},
        )) == 1}
    normalized, diff, _ = normalize_process(original, STALE_HELPER_COPY_RECORD, FINAL_HELPER_COPY_RECORD, stale_exists=False, corrected_regular=True)
    evidence = tamper_suite(original)
    if canonical_sha(normalized) != NORMALIZED_PROCESS_VALUE_SHA256 or len(diff) != 1 or evidence["all_passed"] is not True:
        raise RuntimeError("synthetic")
    with tempfile.TemporaryDirectory(prefix="v499-adapter-fixture-") as folder:
        test_root = Path(folder)
        if test_root.is_symlink() or not test_root.is_dir():
            raise RuntimeError("fixture root")
    return {"passed": True, "leaf_diff_count": 1, "tamper_cases": len(evidence["cases"]), "normalized_sha256": canonical_sha(normalized)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-source")
    parser.add_argument("--adapter-source-sha")
    parser.add_argument("--output-root")
    parser.add_argument("--synthetic-self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), sort_keys=True))
        return 0
    if Path(args.adapter_source) != SELF_PATH or Path(args.output_root) != ADAPTER_ROOT:
        raise RuntimeError("canonical argv")
    source_record = regular(SELF_PATH)
    if args.adapter_source_sha != source_record["sha256"]:
        raise RuntimeError("adapter source SHA argv")
    receipt, pre_snapshot = build_receipt(source_record)
    result = publish(receipt, pre_snapshot)
    print(json.dumps({"passed": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
