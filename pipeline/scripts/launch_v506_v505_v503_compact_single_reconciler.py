#!/usr/bin/env python3
"""Compact exact-once wrapper for the frozen read-only reconciler r2."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS = ROOT / "pipeline/scripts"
RLPY = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
RLPY_INTERMEDIATE = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python")
RLPY_RESOLVED = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
RLPY_SHA = "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788"
RLPY_BYTES = 25555040

WRAPPER = SCRIPTS / "launch_v506_v505_v503_compact_single_reconciler.py"
AUTHORITY_CONTRACT = SCRIPTS / "v506_v505_v503_compact_reconciliation_execution_authority_contract.json"
AUTHORITY_MATERIALIZER = SCRIPTS / "materialize_v506_v505_v503_compact_reconciliation_execution_authority.py"
BASE_AUTHORITY_MATERIALIZER = SCRIPTS / "materialize_v503_v502_compact_reconciliation_execution_authority.py"
BASE_AUTHORITY_MATERIALIZER_SHA = "bfdf33ab525d3b9e98ea116604f309354160cf98021478c94c177caa5060d879"
BASE_AUTHORITY_MATERIALIZER_BYTES = 42234
AUTHORITY_ROOT = J / "v506_v505_v503_compact_reconciliation_execution_authority_seed1647_20260825"
AUTHORITY_RECEIPT = AUTHORITY_ROOT / "authority_receipt.json"
ATTEMPT_ROOT = J / "v506_v505_v503_compact_single_reconciler_attempt_seed1647_20260825"
ATTEMPT_PREP = ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + ".attempt-prep")
R2 = SCRIPTS / "reconcile_v488_v487_c71_exact7_schema_repair.py"
R2_SHA = "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377"
R2_BYTES = 51845
PHASE_CONTRACT = SCRIPTS / "v485_v482_v169_cache_determinism_scope_repair_contract.json"
PHASE_CONTRACT_SHA = "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
PHASE_CONTRACT_BYTES = 43960
REPAIR_FORMAL = J / "v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json"
REPAIR_FORMAL_SHA = "b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b"
REPAIR_FORMAL_BYTES = 36181
STATIC_RECEIPT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json"
STATIC_RECEIPT_SHA = "441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb"
STATIC_RECEIPT_BYTES = 49008
F813 = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json"
F813_SHA = "f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8"
F813_BYTES = 21296
F813_LOG = F813.parent / "immutable_evidence/v485_static_b73.log"
F813_LOG_SHA = "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b"
F813_LOG_BYTES = 6475
TRANSPARENT = F813.parent / "transparent_static_audit.json"
TRANSPARENT_TMP = TRANSPARENT.with_name(TRANSPARENT.name + ".tmp")
QUALIFICATION = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SPLIT_FORENSIC = SCRIPTS / "v506_v505_v503_authority_transport_split_state_forensic.json"
SPLIT_FORENSIC_SHA = "f1a383247039513c8657afe5da8395024200fec66869fb417d459f988821a003"
SPLIT_FORENSIC_BYTES = 7557
V503_AUTHORITY_ROOT = J / "v503_v502_compact_reconciliation_execution_authority_seed1645_20260825"
V503_AUTHORITY_RECEIPT = V503_AUTHORITY_ROOT / "authority_receipt.json"
V503_AUTHORITY_SHA = "394c65ae0ca0979b0ee3c9c774e3b69d68b7ff5e5b1e95e17f02da74c06b223b"
V503_AUTHORITY_BYTES = 46959
V505_EVIDENCE = J / "v505_v503_compact_reconciliation_execution_authority_materialization_evidence_seed1646_20260825"
V505_PROCESS = V505_EVIDENCE / "process_receipt.json"
V505_PROCESS_SHA = "ff2b4b11c47cdd255b61b6941692f2793bf8e6125a0c93b70a8cea2f981fac59"
V505_PROCESS_BYTES = 2982
V505_STDOUT = V505_EVIDENCE / "materializer_stdout.log"
V505_STDOUT_SHA = "1f15250dbb9153126cba342f9166fa2472e00a48d7e6a903f679673741338eae"
V505_STDOUT_BYTES = 1132

CONTRACT_FORMAT = "strict-track2-v506-v505-v503-compact-reconciliation-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
AUTHORITY_FORMAT = "strict-track2-v506-v505-v503-compact-reconciliation-execution-authority-v1"
AUTHORITY_STATUS = "authorized_exact_one_external_v506_compact_single_reconciler_attempt"
INTENT_FORMAT = "strict-track2-v506-v505-v503-compact-single-reconciler-attempt-intent-v1"
TERMINAL_FORMAT = "strict-track2-v506-v505-v503-compact-single-reconciler-attempt-terminal-v1"

CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_role_order",
    "source_aliases", "source_closure_sha256", "authority_materializer_source",
    "base_authority_materializer_source", "repair_formal_record", "postregistration_static_record",
    "phase_a_design_contract_record", "f813_registration_tree", "v505_split_state_forensic",
    "v503_materialized_authority_receipt", "v503_materialized_authority_registration_tree",
    "v505_materialization_evidence_tree", "v505_materialization_process_receipt",
    "v505_materialization_stdout", "v505_split_state_transition", "historical_absences",
    "current_absences_after_authority", "authority_receipt_contract", "authorization",
    "runtime_observation", "execution_boundary",
}
CONTRACT_SOURCE_ORDER = [
    "authority_materializer", "base_authority_materializer", "direct_reconciler_wrapper", "reconciler_r2",
    "phase_a_design_contract", "repair_formal_receipt", "postregistration_static_receipt",
]
CONTRACT_SOURCE_ALIASES = {
    "authority_materializer": "authority_materializer_source",
    "base_authority_materializer": "base_authority_materializer_source",
    "direct_reconciler_wrapper": "direct_reconciler_wrapper_source",
    "reconciler_r2": "reconciler_r2_source",
    "phase_a_design_contract": "phase_a_design_contract_source",
    "repair_formal_receipt": "repair_formal_source",
    "postregistration_static_receipt": "postregistration_static_source",
}
AUTHORITY_SOURCE_ORDER = ["authority_design_contract", *CONTRACT_SOURCE_ORDER]
AUTHORITY_SOURCE_ALIASES = {"authority_design_contract": "authority_design_contract", **CONTRACT_SOURCE_ALIASES}
AUTHORIZATION = {
    "direct_single_reconciler_authorized": True,
    "direct_single_reconciler_invocations_authorized": 1,
    "direct_single_reconciler_invocations_consumed": 0,
    "direct_r2_authorized": False,
    "nested_r2_invocations_authorized": 1,
    "nested_r2_only_via_single_reconciler": True,
    "retry_authorized": False,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "folds_authorized": 0,
    "policy_updates": 0,
    "reward_read_authorized": False,
    "rl_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
    "submission_authorized": False,
}
RUNTIME_BEFORE = {
    "execution_authority_materialized": True,
    "direct_single_reconciler_executed": False,
    "reconciler_r2_executed": False,
    "transparent_receipt_created": False,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True,
    "direct_single_reconciler_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_invocations": 0,
    "training_invocations": 0,
    "reward_reads": 0,
    "dev_hidden_final_outcome_reads": 0,
}
FALSE_BOUNDARY = {
    "phase_a_executed": False,
    "training_launched": False,
    "reward_read": False,
    "dev_hidden_final_outcome_read": False,
    "folds": 0,
    "policy_updates": 0,
    "retry_authorized": False,
}
F813_EXACT2 = {
    "inventory": [
        ["immutable_evidence/v485_static_b73.log", F813_LOG_SHA, F813_LOG_BYTES],
        ["preregistration.json", F813_SHA, F813_BYTES],
    ],
    "file_count": 2,
    "logical_file_bytes": 27771,
    "sha256sum_lines_digest_sha256": "717d6ae12fbacbaafa147d03503140e5f807c028acc97a7294c84da2d79fe3fe",
    "canonical_json_triples_digest_sha256": "a0602666a7e59d60e464b3bc76b31b2fbf501e3d37bbd0b18b2db8c1233971ec",
}
R2_RECEIPT_KEYS = {
    "cache_reuse_authorized", "check_key_set_sha256", "check_keys", "checks", "checks_sha256",
    "contract", "dev_hidden_final_outcome_read_authorized", "f813_registration_tree",
    "false_positive_recomputation", "folds_authorized", "format", "immutable_absences",
    "input_post_snapshot", "input_pre_snapshot", "input_snapshots_exactly_equal",
    "old_reconciler_c71_source", "parent_runtime_observation", "passed",
    "persistent_old_static_log", "phase_a_cache_qualification_authorized", "phase_a_executed",
    "policy_updates", "preregistration", "receipt_writer", "reconciler_r2_source",
    "reconciliation_ancestry", "reconciliation_design_contract", "reconciliation_format",
    "reconciliation_only", "reconciliation_preregistration", "reconciliation_runtime_observation",
    "reconciliation_status", "registration_initial_inventory", "repair_design_contract",
    "repair_formal_ancestry", "repair_format", "repair_materializer_source",
    "repair_preregistration", "repair_registration_tree", "repair_source_closure",
    "reward_read_authorized", "rl_authorized", "runtime_observation", "s1_authorized", "sources",
    "sources_digest_sha256", "static_auditor_self_sha256", "status", "submission_authorized",
    "synthetic_recomputation_evidence", "training_authorized", "v169_imported_or_run",
    "volatile_log_source_at_registration", "zero_update_authorized",
}
R2_CHECK_KEYSET_SHA = "57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b"


class Interrupted(BaseException):
    pass


def cjson(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def csha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: str | Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict:
    path = Path(path)
    metadata = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"not regular nonsymlink: {path}")
    record = {"path": str(path), "sha256": sha256(path), "logical_bytes": metadata.st_size}
    if expected_sha is not None and record["sha256"] != expected_sha:
        raise RuntimeError(f"sha mismatch: {path}")
    if expected_bytes is not None and record["logical_bytes"] != expected_bytes:
        raise RuntimeError(f"size mismatch: {path}")
    return record


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("renameat2 required")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            raise RuntimeError("short write")
        view = view[count:]


def write_noreplace(path: Path, value: object, publish_state: dict | None = None, fault_hook=None) -> dict:
    temporary = path.with_name("." + path.name + ".v506.tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise FileExistsError(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    identity = None
    visible = False
    try:
        identity_metadata = os.fstat(descriptor)
        identity = (identity_metadata.st_dev, identity_metadata.st_ino)
        try:
            write_all(descriptor, cjson(value))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temp_record = regular(temporary)
        os.link(temporary, path, follow_symlinks=False)
        visible = True
        if publish_state is not None:
            publish_state["visible"] = True
            publish_state["record"] = regular(path)
        if fault_hook is not None:
            fault_hook("after_link")
        if (os.lstat(path).st_dev, os.lstat(path).st_ino) != identity:
            raise RuntimeError("publish identity")
        if regular(path)["sha256"] != temp_record["sha256"]:
            raise RuntimeError("publish hash")
        if fault_hook is not None:
            fault_hook("before_unlink")
        os.unlink(temporary)
        identity = None
        if fault_hook is not None:
            fault_hook("before_dir_fsync")
        fsync_directory(path.parent)
        if fault_hook is not None:
            fault_hook("after_dir_fsync")
        record = regular(path)
        if publish_state is not None:
            publish_state["record"] = record
        return record
    except BaseException:
        if identity is not None and os.path.lexists(temporary):
            metadata = os.lstat(temporary)
            if (metadata.st_dev, metadata.st_ino) != identity:
                raise RuntimeError("temporary identity drift")
            os.unlink(temporary)
            fsync_directory(path.parent)
        if visible:
            fsync_directory(path.parent)
        raise


def exact_tree(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root: {root}")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = os.lstat(path)
        if path.is_symlink():
            raise RuntimeError(f"tree member: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"tree member: {path}")
        inventory.append([path.relative_to(root).as_posix(), sha256(path), metadata.st_size])
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in inventory).encode()
    return {
        "root": str(root), "inventory": inventory, "file_count": len(inventory),
        "logical_file_bytes": sum(row[2] for row in inventory),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": csha(inventory),
    }


def tree_body(tree: dict) -> dict:
    return {key: value for key, value in tree.items() if key != "root"}


def exact_f813_before() -> dict:
    tree = exact_tree(F813.parent)
    if tree_body(tree) != F813_EXACT2:
        raise RuntimeError("F813 is not frozen exact2")
    regular(F813_LOG, F813_LOG_SHA, F813_LOG_BYTES)
    regular(F813, F813_SHA, F813_BYTES)
    return tree


def source_records() -> dict:
    return {
        "base_authority_materializer": regular(
            BASE_AUTHORITY_MATERIALIZER, BASE_AUTHORITY_MATERIALIZER_SHA,
            BASE_AUTHORITY_MATERIALIZER_BYTES),
        "reconciler_r2": regular(R2, R2_SHA, R2_BYTES),
        "phase_a_design_contract": regular(PHASE_CONTRACT, PHASE_CONTRACT_SHA, PHASE_CONTRACT_BYTES),
        "repair_formal_receipt": regular(REPAIR_FORMAL, REPAIR_FORMAL_SHA, REPAIR_FORMAL_BYTES),
        "postregistration_static_receipt": regular(STATIC_RECEIPT, STATIC_RECEIPT_SHA, STATIC_RECEIPT_BYTES),
    }


def ancestor_pids(pid: int) -> set[int]:
    result = {pid}
    current = pid
    while current > 1:
        try:
            lines = (Path("/proc") / str(current) / "status").read_text().splitlines()
            parent = int(next(line.split(":", 1)[1] for line in lines if line.startswith("PPid:")))
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            break
        if parent <= 0 or parent in result:
            break
        result.add(parent)
        current = parent
    return result


def live_lineage_processes() -> list[dict]:
    excluded = ancestor_pids(os.getpid())
    entrypoints = {str(WRAPPER), str(R2), str(AUTHORITY_MATERIALIZER)}
    rows = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) in excluded:
            continue
        try:
            argv = [part.decode("utf-8", "replace") for part in (process_dir / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(argv) >= 2 and argv[0] == str(RLPY) and argv[1] in entrypoints:
            rows.append({"pid": int(process_dir.name), "argv": argv})
    return sorted(rows, key=lambda row: row["pid"])


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi rc={result.returncode}: {result.stderr}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def validate_split_state_ancestry(contract: dict, authority: dict) -> dict:
    split_record = regular(SPLIT_FORENSIC, SPLIT_FORENSIC_SHA, SPLIT_FORENSIC_BYTES)
    split = json.loads(SPLIT_FORENSIC.read_text())
    expected_top = {
        "format", "status", "passed", "date", "prior_failure_forensic", "state_transition",
        "authority_receipt", "authority_registration_tree", "authority_schema",
        "v505_materialization_evidence_tree", "v505_process_receipt", "v505_process_schema",
        "stdout_record", "stdout_contract", "invocation_partition", "runtime_absences",
        "post_observation", "authorization",
    }
    v503_authority_record = regular(V503_AUTHORITY_RECEIPT, V503_AUTHORITY_SHA, V503_AUTHORITY_BYTES)
    v503_authority_tree = exact_tree(V503_AUTHORITY_ROOT)
    v503_authority = json.loads(V503_AUTHORITY_RECEIPT.read_text())
    v505_evidence_tree = exact_tree(V505_EVIDENCE)
    v505_process_record = regular(V505_PROCESS, V505_PROCESS_SHA, V505_PROCESS_BYTES)
    v505_process = json.loads(V505_PROCESS.read_text())
    stdout_record = regular(V505_STDOUT, V505_STDOUT_SHA, V505_STDOUT_BYTES)
    stdout_payload = V505_STDOUT.read_bytes()
    stdout_lines_raw = stdout_payload.splitlines(keepends=True)
    stdout_lines = [json.loads(line) for line in stdout_lines_raw]
    transition = split.get("state_transition", {})
    contract_transition = {
        "authority_materialized": True,
        "authority_receipt_passed": True,
        "transport_helper_terminal_passed": False,
        "transport_helper_failure_stage": "post_authority_snapshot_revalidated_pre_authority_absence",
        "authority_materializer_invocations": 1,
        "direct_single_reconciler_invocations": 0,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "fresh_authority_required": True,
    }
    partition = split.get("invocation_partition", {})
    process_schema = split.get("v505_process_schema", {})
    stdout_contract = split.get("stdout_contract", {})
    if (set(split) != expected_top or split.get("passed") is not True
            or split.get("status") != "authority_materialized_exact_once_transport_failed_no_wrapper_execution"
            or transition != {
                "predicate_name": "prior_failure_forensic.runtime_roots.authority_root.absent",
                "prior_expected": True, "prior_observed": True, "current_observed": False,
                "transition_authorized": True,
                "transition_cause": "bfdf materializer successfully committed the v503 authority registration exact1",
                "sole_changed_runtime_root_predicate": True,
                "other_prior_forensic_predicates_still_true": True}
            or split.get("authority_receipt") != v503_authority_record
            or split.get("authority_registration_tree") != v503_authority_tree
            or split.get("v505_materialization_evidence_tree") != v505_evidence_tree
            or split.get("v505_process_receipt") != v505_process_record
            or split.get("stdout_record") != stdout_record):
        raise RuntimeError("split state primary anchors")
    if (len(v503_authority) != 46 or len(v503_authority.get("checks", {})) != 33
            or v503_authority.get("passed") is not True
            or split.get("authority_schema", {}).get("top_key_set_sha256") != csha(sorted(v503_authority))
            or split.get("authority_schema", {}).get("checks_sha256") != csha(v503_authority["checks"])
            or split.get("authority_schema", {}).get("source_closure_sha256") != v503_authority.get("source_closure_sha256")):
        raise RuntimeError("split state authority schema")
    if (process_schema.get("status") != "failed_no_retry" or process_schema.get("passed") is not False
            or process_schema.get("key_count") != len(v505_process)
            or process_schema.get("key_set_sha256") != csha(sorted(v505_process))
            or process_schema.get("error") != v505_process.get("error")
            or process_schema.get("cleanup") != v505_process.get("cleanup")
            or v505_process.get("authority_materializer_invocations") != 1
            or v505_process.get("transport_helper_invocations") != 1
            or v505_process.get("direct_single_reconciler_invocations") != 0
            or v505_process.get("reconciler_r2_invocations") != 0
            or v505_process.get("retry_authorized") is not False):
        raise RuntimeError("split state process schema")
    if (len(stdout_lines) != 2 or b"".join(stdout_lines_raw) != stdout_payload
            or any(cbytes(value) != raw for value, raw in zip(stdout_lines, stdout_lines_raw))
            or stdout_contract.get("line_count") != 2
            or stdout_contract.get("line_1", {}).get("sha256") != hashlib.sha256(stdout_lines_raw[0]).hexdigest()
            or stdout_contract.get("line_1", {}).get("logical_bytes") != len(stdout_lines_raw[0])
            or stdout_contract.get("line_2", {}).get("sha256") != hashlib.sha256(stdout_lines_raw[1]).hexdigest()
            or stdout_contract.get("line_2", {}).get("logical_bytes") != len(stdout_lines_raw[1])
            or stdout_lines[0].get("materializer_native_empty") is not True
            or stdout_lines[1].get("committed_success") is not True
            or stdout_lines[1].get("passed") is not True):
        raise RuntimeError("split state stdout exact2")
    if (partition != {"v503_failed_transport_helper_invocations": 1,
                      "v505_transport_helper_invocations": 1,
                      "authority_materializer_invocations": 1,
                      "direct_single_reconciler_invocations": 0,
                      "reconciler_r2_invocations": 0,
                      "phase_a_invocations": 0, "training_invocations": 0}
            or not all(not os.path.lexists(path) for path in split.get("runtime_absences", {}).values())
            or split.get("post_observation") != {"all_runtime_absences_true": True,
                                                  "lineage_processes_empty": True,
                                                  "gpu_compute_processes_empty": True}):
        raise RuntimeError("split state partition/absences")
    return {"split_state_forensic": split_record, "v503_authority_receipt": v503_authority_record,
            "v503_authority_registration_tree": v503_authority_tree,
            "v505_materialization_evidence_tree": v505_evidence_tree,
            "v505_process_receipt": v505_process_record, "v505_stdout": stdout_record,
            "forensic_transition": transition, "contract_transition": contract_transition}


def validate_compact_authority(args: argparse.Namespace) -> dict:
    if (not RLPY.is_symlink() or os.readlink(RLPY) != str(RLPY_INTERMEDIATE)
            or not RLPY_INTERMEDIATE.is_symlink() or os.readlink(RLPY_INTERMEDIATE) != "python3.11"
            or RLPY.resolve() != RLPY_RESOLVED
            or regular(RLPY_RESOLVED, RLPY_SHA, RLPY_BYTES)["path"] != str(RLPY_RESOLVED)
            or Path(sys.executable) != RLPY):
        raise RuntimeError("execution interpreter chain")
    if args.authority_contract.resolve() != AUTHORITY_CONTRACT or args.authority_receipt.resolve() != AUTHORITY_RECEIPT:
        raise RuntimeError("authority paths")
    if args.wrapper_source.resolve() != WRAPPER or args.wrapper_source.resolve() != Path(__file__).resolve():
        raise RuntimeError("wrapper path")
    contract_record = regular(args.authority_contract, args.authority_contract_sha)
    authority_record = regular(args.authority_receipt, args.authority_receipt_sha)
    wrapper_record = regular(args.wrapper_source, args.wrapper_sha)
    contract = json.loads(args.authority_contract.read_text())
    authority = json.loads(args.authority_receipt.read_text())
    if set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1647:
        raise RuntimeError("compact contract schema")
    if contract.get("authorization") != AUTHORIZATION or contract.get("runtime_observation") != RUNTIME_BEFORE or contract.get("execution_boundary") != EXECUTION_BOUNDARY:
        raise RuntimeError("compact contract authorization")
    receipt_contract = contract.get("authority_receipt_contract")
    if (not isinstance(receipt_contract, dict)
            or set(receipt_contract) != {"format", "status", "top_keys", "check_keys", "check_key_set_sha256", "checks_sha256", "authorization_exact", "runtime_observation_exact"}
            or set(authority) != set(receipt_contract.get("top_keys", []))
            or receipt_contract.get("format") != AUTHORITY_FORMAT
            or receipt_contract.get("status") != AUTHORITY_STATUS
            or receipt_contract.get("authorization_exact") != AUTHORIZATION
            or receipt_contract.get("runtime_observation_exact") != RUNTIME_BEFORE
            or receipt_contract.get("check_keys") != authority.get("check_keys")
            or receipt_contract.get("check_key_set_sha256") != authority.get("check_key_set_sha256")
            or receipt_contract.get("checks_sha256") != authority.get("checks_sha256")):
        raise RuntimeError("compact authority top schema")
    if authority.get("format") != AUTHORITY_FORMAT or authority.get("status") != AUTHORITY_STATUS or authority.get("passed") is not True:
        raise RuntimeError("compact authority status")
    check_keys = authority.get("check_keys")
    checks = authority.get("checks")
    if not isinstance(check_keys, list) or check_keys != sorted(checks) or checks != {key: True for key in check_keys}:
        raise RuntimeError("compact authority checks")
    if authority.get("check_key_set_sha256") != csha(check_keys) or authority.get("checks_sha256") != csha(checks):
        raise RuntimeError("compact authority check digests")
    if authority.get("authorization") != AUTHORIZATION or authority.get("runtime_observation") != RUNTIME_BEFORE or authority.get("execution_boundary") != EXECUTION_BOUNDARY:
        raise RuntimeError("compact authority boundary")
    if (authority.get("input_snapshots_exactly_equal") is not True
            or authority.get("input_pre_snapshot") != authority.get("input_post_snapshot")):
        raise RuntimeError("compact authority input snapshots")
    if authority.get("authority_design_contract") != contract_record:
        raise RuntimeError("compact contract alias")

    contract_sources = contract.get("source_closure")
    if contract.get("source_role_order") != CONTRACT_SOURCE_ORDER or contract.get("source_aliases") != CONTRACT_SOURCE_ALIASES:
        raise RuntimeError("compact source schema")
    if (not isinstance(contract_sources, dict) or set(contract_sources) != set(CONTRACT_SOURCE_ORDER)
            or contract.get("source_closure_sha256") != csha(contract_sources)):
        raise RuntimeError("compact source closure")
    observed = {role: regular(row["path"], row["sha256"], row["logical_bytes"]) for role, row in contract_sources.items()}
    if observed != contract_sources:
        raise RuntimeError("compact source drift")
    if observed["direct_reconciler_wrapper"] != wrapper_record:
        raise RuntimeError("compact wrapper alias")
    if observed["authority_materializer"]["path"] != str(AUTHORITY_MATERIALIZER):
        raise RuntimeError("compact materializer path")
    if contract.get("authority_materializer_source") != observed["authority_materializer"] or authority.get("authority_materializer_source") != observed["authority_materializer"]:
        raise RuntimeError("compact materializer alias")
    expected_fixed = source_records()
    if any(observed[role] != record for role, record in expected_fixed.items()):
        raise RuntimeError("compact fixed source records")
    if (contract.get("base_authority_materializer_source") != expected_fixed["base_authority_materializer"]
            or authority.get("base_authority_materializer_source") != expected_fixed["base_authority_materializer"]):
        raise RuntimeError("compact base materializer alias")
    if exact_tree(REPAIR_FORMAL.parent)["inventory"] != [[REPAIR_FORMAL.name, REPAIR_FORMAL_SHA, REPAIR_FORMAL_BYTES]]:
        raise RuntimeError("repair registration exact1")
    if exact_tree(STATIC_RECEIPT.parent)["inventory"] != [[STATIC_RECEIPT.name, STATIC_RECEIPT_SHA, STATIC_RECEIPT_BYTES]]:
        raise RuntimeError("static registration exact1")
    for role, alias in CONTRACT_SOURCE_ALIASES.items():
        if authority.get(alias) != observed[role]:
            raise RuntimeError(f"compact source alias: {role}")
    authority_sources = authority.get("source_closure")
    if (not isinstance(authority_sources, dict) or set(authority_sources) != set(AUTHORITY_SOURCE_ORDER)
            or authority.get("source_role_order") != AUTHORITY_SOURCE_ORDER
            or authority.get("source_aliases") != AUTHORITY_SOURCE_ALIASES):
        raise RuntimeError("compact authority source order")
    if authority_sources != {"authority_design_contract": contract_record, **observed} or authority.get("source_closure_sha256") != csha(authority_sources):
        raise RuntimeError("compact authority source closure")

    repair_record = expected_fixed["repair_formal_receipt"]
    static_record = expected_fixed["postregistration_static_receipt"]
    phase_record = expected_fixed["phase_a_design_contract"]
    if contract.get("repair_formal_record") != repair_record or authority.get("repair_formal_source") != repair_record:
        raise RuntimeError("repair alias")
    if contract.get("postregistration_static_record") != static_record or authority.get("postregistration_static_source") != static_record:
        raise RuntimeError("static alias")
    if contract.get("phase_a_design_contract_record") != phase_record or authority.get("phase_a_design_contract_source") != phase_record:
        raise RuntimeError("phase alias")

    f813_tree = exact_f813_before()
    if contract.get("f813_registration_tree") != f813_tree or authority.get("f813_registration_tree") != f813_tree:
        raise RuntimeError("F813 authority alias")
    anchors = {}
    split_ancestry = validate_split_state_ancestry(contract, authority)
    split_aliases = {
        "v505_split_state_forensic": split_ancestry["split_state_forensic"],
        "v503_materialized_authority_receipt": split_ancestry["v503_authority_receipt"],
        "v503_materialized_authority_registration_tree": split_ancestry["v503_authority_registration_tree"],
        "v505_materialization_evidence_tree": split_ancestry["v505_materialization_evidence_tree"],
        "v505_materialization_process_receipt": split_ancestry["v505_process_receipt"],
        "v505_materialization_stdout": split_ancestry["v505_stdout"],
        "v505_split_state_transition": split_ancestry["contract_transition"],
    }
    if any(contract.get(key) != value or authority.get(key) != value
           for key, value in split_aliases.items()):
        raise RuntimeError("split state contract/authority aliases")
    lineage = {
        "fresh_authority_root": str(AUTHORITY_ROOT),
        "fresh_wrapper_attempt_root": str(ATTEMPT_ROOT),
        "name": "v506_v505_v503_compact_reconciliation",
        "supersedes_v505_split_authority": True,
    }
    if contract.get("lineage") != lineage:
        raise RuntimeError("compact lineage")
    if authority.get("wrapper_attempt_root") != str(ATTEMPT_ROOT) or authority.get("transparent_static_receipt_path") != str(TRANSPARENT):
        raise RuntimeError("compact output aliases")
    historical_contract = contract.get("historical_absences")
    current_contract = contract.get("current_absences_after_authority")
    historical_authority = authority.get("historical_absences")
    current_authority = authority.get("required_absences")
    if not all(isinstance(rows, dict) for rows in (historical_contract, current_contract, historical_authority, current_authority)):
        raise RuntimeError("compact absence schema")
    if ({key: row.get("path") for key, row in historical_authority.items()}
            != {key: row.get("path") for key, row in historical_contract.items()}):
        raise RuntimeError("compact historical absences")
    if any(row.get("absent") is not True for row in historical_authority.values()):
        raise RuntimeError("compact historical absence declarations")
    if ({key: row.get("path") for key, row in current_authority.items()}
            != {key: row.get("path") for key, row in current_contract.items()}):
        raise RuntimeError("compact current absences")
    if any(row.get("absent") is not True or os.path.lexists(row.get("path")) for row in current_authority.values()):
        raise RuntimeError("compact live absences")
    if any(os.path.lexists(path) for path in (ATTEMPT_ROOT, ATTEMPT_PREP, TRANSPARENT, TRANSPARENT_TMP, QUALIFICATION)):
        raise RuntimeError("compact output prestate")
    if live_lineage_processes() or gpu_processes():
        raise RuntimeError("compact live process prestate")
    return {
        "contract": contract, "authority": authority, "contract_record": contract_record,
        "authority_record": authority_record, "wrapper_record": wrapper_record,
        "sources": observed, "anchors": anchors, "split_ancestry": split_ancestry,
        "f813_before": f813_tree,
    }


def derive_r2_command(context: dict) -> list[str]:
    repair = json.loads(REPAIR_FORMAL.read_text())
    f813 = json.loads(F813.read_text())
    static = json.loads(STATIC_RECEIPT.read_text())
    if static.get("passed") is not True:
        raise RuntimeError("repair/static status")
    repair_sources = repair.get("source_closure")
    frozen_parent = f813.get("frozen_parent")
    if not isinstance(repair_sources, dict) or not isinstance(frozen_parent, dict):
        raise RuntimeError("repair/F813 schema")
    phase_contract = frozen_parent["phase_a_design_contract"]
    old_source = frozen_parent["old_static_source"]
    old_receipt = frozen_parent["old_failed_static_receipt"]
    old_log = frozen_parent["old_static_log_volatile_source"]
    reconciliation_contract = f813["design_contract"]
    parent_preregistration = repair_sources["parent_v485_preregistration"]
    for row in (phase_contract, old_source, old_receipt, old_log, reconciliation_contract, parent_preregistration):
        regular(row["path"], row["sha256"], row.get("logical_bytes"))
    if {key: phase_contract[key] for key in ("path", "sha256")} != {key: context["sources"]["phase_a_design_contract"][key] for key in ("path", "sha256")}:
        raise RuntimeError("phase command binding")
    return [
        str(RLPY), str(R2),
        "--preregistration", parent_preregistration["path"],
        "--preregistration-sha", parent_preregistration["sha256"],
        "--contract", phase_contract["path"], "--contract-sha", phase_contract["sha256"],
        "--old-static-source", old_source["path"], "--old-static-sha", old_source["sha256"],
        "--old-static-receipt", old_receipt["path"], "--old-static-receipt-sha", old_receipt["sha256"],
        "--old-static-log", old_log["path"], "--old-static-log-sha", old_log["sha256"],
        "--reconciliation-preregistration", str(F813), "--reconciliation-preregistration-sha", F813_SHA,
        "--repair-preregistration", str(REPAIR_FORMAL), "--repair-preregistration-sha", REPAIR_FORMAL_SHA,
        "--reconciliation-contract", reconciliation_contract["path"], "--reconciliation-contract-sha", reconciliation_contract["sha256"],
        "--reconciler-source", str(R2), "--reconciler-sha", R2_SHA,
        "--output", str(TRANSPARENT),
    ]


def immutable_snapshot(context: dict) -> dict:
    sources = {role: regular(row["path"], row["sha256"], row["logical_bytes"]) for role, row in context["sources"].items()}
    files = {
        "contract": regular(context["contract_record"]["path"], context["contract_record"]["sha256"], context["contract_record"]["logical_bytes"]),
        "authority": regular(context["authority_record"]["path"], context["authority_record"]["sha256"], context["authority_record"]["logical_bytes"]),
        "wrapper": regular(context["wrapper_record"]["path"], context["wrapper_record"]["sha256"], context["wrapper_record"]["logical_bytes"]),
        "v505_split_state_forensic": regular(SPLIT_FORENSIC, SPLIT_FORENSIC_SHA, SPLIT_FORENSIC_BYTES),
        "v503_materialized_authority": regular(V503_AUTHORITY_RECEIPT, V503_AUTHORITY_SHA, V503_AUTHORITY_BYTES),
        "v505_materialization_process": regular(V505_PROCESS, V505_PROCESS_SHA, V505_PROCESS_BYTES),
        "v505_materialization_stdout": regular(V505_STDOUT, V505_STDOUT_SHA, V505_STDOUT_BYTES),
        "f813_preregistration": regular(F813, F813_SHA, F813_BYTES),
        "f813_log": regular(F813_LOG, F813_LOG_SHA, F813_LOG_BYTES),
    }
    trees = {
        "v503_materialized_authority": exact_tree(V503_AUTHORITY_ROOT),
        "v505_materialization_evidence": exact_tree(V505_EVIDENCE),
        "repair": exact_tree(REPAIR_FORMAL.parent),
        "static": exact_tree(STATIC_RECEIPT.parent),
    }
    return {"files": files, "sources": sources, "trees": trees}


def commit_intent(intent: dict, fault_hook=None):
    baseline = None
    identity = None
    stdout_stream = None
    stderr_stream = None
    blocked = False
    promoted = False
    owned_members = {}
    try:
        baseline = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        blocked = True
        ATTEMPT_PREP.mkdir()
        metadata = os.lstat(ATTEMPT_PREP)
        identity = (metadata.st_dev, metadata.st_ino)
        intent_record = write_noreplace(ATTEMPT_PREP / "intent.json", intent)
        intent_metadata = os.lstat(ATTEMPT_PREP / "intent.json")
        owned_members["intent.json"] = {
            "identity": (intent_metadata.st_dev, intent_metadata.st_ino), "record": intent_record,
        }
        stdout_stream = (ATTEMPT_PREP / "reconciler_stdout.log").open("xb", buffering=0)
        stderr_stream = (ATTEMPT_PREP / "reconciler_stderr.log").open("xb", buffering=0)
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
        for name in ("reconciler_stdout.log", "reconciler_stderr.log"):
            path = ATTEMPT_PREP / name
            member_metadata = os.lstat(path)
            owned_members[name] = {
                "identity": (member_metadata.st_dev, member_metadata.st_ino), "record": regular(path),
            }
        fsync_directory(ATTEMPT_PREP)
        if [path.name for path in sorted(ATTEMPT_PREP.iterdir())] != [
            "intent.json", "reconciler_stderr.log", "reconciler_stdout.log",
        ]:
            raise RuntimeError("attempt prep exact3")
        if fault_hook is not None:
            fault_hook(ATTEMPT_PREP)
        if (os.lstat(ATTEMPT_PREP).st_dev, os.lstat(ATTEMPT_PREP).st_ino) != identity or os.path.lexists(ATTEMPT_ROOT):
            raise RuntimeError("attempt prep ownership")
        rename_directory_noreplace(ATTEMPT_PREP, ATTEMPT_ROOT)
        promoted = True
        fsync_directory(ATTEMPT_ROOT.parent)
        promoted = os.lstat(ATTEMPT_ROOT)
        if (promoted.st_dev, promoted.st_ino) != identity:
            raise RuntimeError("attempt promoted identity")
        if [path.name for path in sorted(ATTEMPT_ROOT.iterdir())] != [
            "intent.json", "reconciler_stderr.log", "reconciler_stdout.log",
        ] or exact_tree(ATTEMPT_ROOT)["file_count"] != 3:
            raise RuntimeError("attempt promoted exact3")
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
        blocked = False
        return stdout_stream, stderr_stream
    except BaseException:
        for stream in (stdout_stream, stderr_stream):
            if stream is not None and not stream.closed:
                stream.close()
        if identity is not None and not promoted and ATTEMPT_PREP.is_dir() and not ATTEMPT_PREP.is_symlink():
            metadata = os.lstat(ATTEMPT_PREP)
            if (metadata.st_dev, metadata.st_ino) != identity:
                raise RuntimeError("attempt prep identity drift; refusing cleanup")
            allowed = {"intent.json", "reconciler_stdout.log", "reconciler_stderr.log"}
            members = {path.name for path in ATTEMPT_PREP.iterdir()}
            if not members.issubset(allowed):
                raise RuntimeError("attempt prep foreign member; refusing cleanup")
            if members != set(owned_members):
                raise RuntimeError("attempt prep owned member set drift; refusing cleanup")
            for name, frozen in owned_members.items():
                path = ATTEMPT_PREP / name
                if path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
                    raise RuntimeError("attempt prep foreign type; refusing cleanup")
                current_metadata = os.lstat(path)
                current_record = regular(path)
                if ((current_metadata.st_dev, current_metadata.st_ino) != frozen["identity"]
                        or current_record != frozen["record"]):
                    raise RuntimeError("attempt prep owned member drift; refusing cleanup")
            for path in ATTEMPT_PREP.iterdir():
                path.unlink()
            ATTEMPT_PREP.rmdir()
            fsync_directory(ATTEMPT_PREP.parent)
        raise
    finally:
        if blocked and baseline is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline)


def spawn_owned(command: list[str], stdout_stream, stderr_stream, state: dict):
    baseline = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    blocked = True
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stdout_stream, stderr=stderr_stream,
            start_new_session=True, close_fds=True,
        )
        state["process"] = process
        state["started"] = True
        state["pgid"] = process.pid
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
        blocked = False
        return process
    finally:
        if blocked:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline)


def close_durable(stream) -> None:
    if stream is None or stream.closed:
        return
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()


def group_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def cleanup_process(process: subprocess.Popen | None) -> dict:
    if process is None:
        return {"started": False, "reaped": True, "group_empty": True, "returncode": None}
    pgid = process.pid
    if process.poll() is None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    else:
        process.wait()
    for _ in range(50):
        if group_empty(pgid):
            break
        time.sleep(0.02)
    return {"started": True, "reaped": process.poll() is not None, "group_empty": group_empty(pgid), "returncode": process.returncode, "pgid": pgid}


def validate_transparent(context: dict) -> tuple[dict, dict, dict]:
    output_record = regular(TRANSPARENT)
    receipt = json.loads(TRANSPARENT.read_text())
    if set(receipt) != R2_RECEIPT_KEYS or receipt.get("format") != "strict-track2-v485-v169-cache-qualification-static-audit-v1" or receipt.get("status") != "passed_no_execution_authority" or receipt.get("passed") is not True:
        raise RuntimeError("transparent schema")
    check_keys = receipt.get("check_keys")
    checks = receipt.get("checks")
    if not isinstance(check_keys, list) or len(check_keys) != 47 or check_keys != sorted(checks) or checks != {key: True for key in check_keys}:
        raise RuntimeError("transparent checks")
    if csha(check_keys) != R2_CHECK_KEYSET_SHA or receipt.get("check_key_set_sha256") != R2_CHECK_KEYSET_SHA or receipt.get("checks_sha256") != csha(checks):
        raise RuntimeError("transparent check digests")
    if receipt.get("repair_preregistration") != context["sources"]["repair_formal_receipt"]:
        raise RuntimeError("transparent repair")
    r2_record = context["sources"]["reconciler_r2"]
    if receipt.get("reconciler_r2_source") != r2_record or receipt.get("receipt_writer") != r2_record:
        raise RuntimeError("transparent reconciler")
    if receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot") != receipt.get("input_post_snapshot"):
        raise RuntimeError("transparent input snapshots")
    if receipt.get("f813_registration_tree") != F813_EXACT2 or receipt.get("registration_initial_inventory") != F813_EXACT2:
        raise RuntimeError("transparent initial F813 tree")
    if receipt.get("reconciliation_preregistration") != {"path": str(F813), "sha256": F813_SHA}:
        raise RuntimeError("transparent F813 alias")
    if receipt.get("phase_a_executed") is not False or receipt.get("v169_imported_or_run") is not False or receipt.get("reconciliation_only") is not True:
        raise RuntimeError("transparent runtime")
    for key in (
        "phase_a_cache_qualification_authorized", "cache_reuse_authorized", "training_authorized",
        "s1_authorized", "zero_update_authorized", "rl_authorized", "submission_authorized",
        "reward_read_authorized", "dev_hidden_final_outcome_read_authorized",
    ):
        if receipt.get(key) is not False:
            raise RuntimeError(f"transparent unsafe: {key}")
    if receipt.get("folds_authorized") != 0 or receipt.get("policy_updates") != 0:
        raise RuntimeError("transparent unsafe counts")
    after = exact_tree(F813.parent)
    expected_inventory = [*F813_EXACT2["inventory"], [TRANSPARENT.name, output_record["sha256"], output_record["logical_bytes"]]]
    if after["inventory"] != expected_inventory or after["file_count"] != 3 or after["logical_file_bytes"] != 27771 + output_record["logical_bytes"]:
        raise RuntimeError("F813 exact2 to sole exact3")
    if os.path.lexists(TRANSPARENT_TMP) or os.path.lexists(QUALIFICATION):
        raise RuntimeError("transparent forbidden outputs")
    return receipt, after, output_record


def commit_terminal(terminal: dict, state: dict, expected_snapshot: dict, snapshot_fn=immutable_snapshot, fault_hook=None) -> dict:
    baseline = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    blocked = True
    try:
        if snapshot_fn(state["context"]) != expected_snapshot:
            raise RuntimeError("immutable input drift before terminal")
        if state.get("visible") is not False or state.get("committed_exact4") is not False:
            raise RuntimeError("terminal state not fresh")
        if [path.name for path in sorted(ATTEMPT_ROOT.iterdir())] != [
            "intent.json", "reconciler_stderr.log", "reconciler_stdout.log",
        ] or exact_tree(ATTEMPT_ROOT)["file_count"] != 3:
            raise RuntimeError("terminal prepublish exact3")
        record = write_noreplace(
            ATTEMPT_ROOT / "terminal_receipt.json", terminal, state, fault_hook,
        )
        if [path.name for path in sorted(ATTEMPT_ROOT.iterdir())] != [
            "intent.json", "reconciler_stderr.log", "reconciler_stdout.log", "terminal_receipt.json",
        ]:
            raise RuntimeError("terminal direct members exact4")
        tree = exact_tree(ATTEMPT_ROOT)
        if tree["file_count"] != 4 or [row[0] for row in tree["inventory"]] != [
            "intent.json", "reconciler_stderr.log", "reconciler_stdout.log", "terminal_receipt.json",
        ]:
            raise RuntimeError("terminal exact4")
        regular(record["path"], record["sha256"], record["logical_bytes"])
        state["committed_exact4"] = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
        blocked = False
        return record
    finally:
        if blocked:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-contract", required=True, type=Path)
    parser.add_argument("--authority-contract-sha", required=True)
    parser.add_argument("--authority-receipt", required=True, type=Path)
    parser.add_argument("--authority-receipt-sha", required=True)
    parser.add_argument("--wrapper-source", required=True, type=Path)
    parser.add_argument("--wrapper-sha", required=True)
    parser.add_argument("--synthetic-self-test", action="store_true")
    return parser.parse_args()


def execute(args: argparse.Namespace) -> int:
    started_ns = time.monotonic_ns()
    stdout_stream = None
    stderr_stream = None
    process_state = {"process": None, "started": False, "pgid": None}
    terminal_state = {"visible": False, "committed_exact4": False, "record": None, "context": None}
    old_handlers = {}
    lock_fd = os.open(args.authority_receipt, os.O_RDONLY)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(signum, _frame):
        raise Interrupted(f"signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        context = validate_compact_authority(args)
        terminal_state["context"] = context
        command = derive_r2_command(context)
        snapshot = immutable_snapshot(context)
        intent = {
            "format": INTENT_FORMAT,
            "status": "committed_before_reconciler_r2_start",
            "passed": True,
            "attempt_nonce": os.urandom(32).hex(),
            "created_epoch_ns": time.time_ns(),
            "command_argv": command,
            "command_argv_sha256": csha(command),
            "authority_receipt": context["authority_record"],
            "wrapper_source": context["wrapper_record"],
            "reconciler_r2_source": context["sources"]["reconciler_r2"],
            "split_state_forensic": context["split_ancestry"]["split_state_forensic"],
            "v503_materialized_authority_receipt": context["split_ancestry"]["v503_authority_receipt"],
            "v505_failed_transport_process": context["split_ancestry"]["v505_process_receipt"],
            "immutable_input_snapshot": snapshot,
            "nested_r2_invocations_before": 0,
            **FALSE_BOUNDARY,
        }
        stdout_stream, stderr_stream = commit_intent(intent)
        process = spawn_owned(command, stdout_stream, stderr_stream, process_state)
        try:
            returncode = process.wait(timeout=300)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("reconciler r2 timeout") from error
        close_durable(stdout_stream)
        close_durable(stderr_stream)
        cleanup = cleanup_process(process)
        if returncode != 0 or cleanup["reaped"] is not True or cleanup["group_empty"] is not True:
            raise RuntimeError(f"reconciler r2 failed: rc={returncode}, cleanup={cleanup}")
        transparent_receipt, f813_after, output_record = validate_transparent(context)
        post_snapshot = immutable_snapshot(context)
        if post_snapshot != snapshot:
            raise RuntimeError("immutable input drift")
        terminal = {
            "format": TERMINAL_FORMAT,
            "status": "passed_exact_one_nested_r2_readonly_reconciliation",
            "passed": True,
            "attempt_nonce": intent["attempt_nonce"],
            "wall_seconds": (time.monotonic_ns() - started_ns) / 1e9,
            "intent": regular(ATTEMPT_ROOT / "intent.json"),
            "authority_receipt": context["authority_record"],
            "wrapper_source": context["wrapper_record"],
            "reconciler_r2_source": context["sources"]["reconciler_r2"],
            "nested_r2_invocations": 1,
            "reconciler_exit_code": 0,
            "cleanup": cleanup,
            "reconciler_stdout": regular(ATTEMPT_ROOT / "reconciler_stdout.log"),
            "reconciler_stderr": regular(ATTEMPT_ROOT / "reconciler_stderr.log"),
            "transparent_static_receipt": output_record,
            "transparent_checks_sha256": transparent_receipt["checks_sha256"],
            "f813_registration_tree_before": context["f813_before"],
            "f813_registration_tree_after": f813_after,
            "f813_exact2_to_sole_exact3": True,
            "split_state_forensic": context["split_ancestry"]["split_state_forensic"],
            "v503_materialized_authority_receipt": context["split_ancestry"]["v503_authority_receipt"],
            "v505_failed_transport_process": context["split_ancestry"]["v505_process_receipt"],
            "immutable_input_pre_snapshot": snapshot,
            "immutable_input_post_snapshot": post_snapshot,
            "immutable_inputs_exactly_equal": True,
            **FALSE_BOUNDARY,
        }
        if not math.isfinite(terminal["wall_seconds"]) or terminal["wall_seconds"] < 0:
            raise RuntimeError("terminal wall")
        commit_terminal(terminal, terminal_state, snapshot)
        return 0
    except BaseException as error:
        try:
            close_durable(stdout_stream)
        except BaseException:
            pass
        try:
            close_durable(stderr_stream)
        except BaseException:
            pass
        try:
            cleanup = cleanup_process(process_state["process"])
        except BaseException as cleanup_error:
            cleanup = {"reaped": False, "group_empty": False, "error": repr(cleanup_error)}
        if (ATTEMPT_ROOT.is_dir() and not ATTEMPT_ROOT.is_symlink()
                and not terminal_state["visible"]
                and not os.path.lexists(ATTEMPT_ROOT / "terminal_receipt.json")):
            failure = {
                "format": TERMINAL_FORMAT,
                "status": "failed_no_retry",
                "passed": False,
                "wall_seconds": (time.monotonic_ns() - started_ns) / 1e9,
                "intent": regular(ATTEMPT_ROOT / "intent.json") if (ATTEMPT_ROOT / "intent.json").is_file() else None,
                "error_type": type(error).__name__,
                "error": str(error),
                "cleanup": cleanup,
                "reconciler_stdout": regular(ATTEMPT_ROOT / "reconciler_stdout.log") if (ATTEMPT_ROOT / "reconciler_stdout.log").is_file() else None,
                "reconciler_stderr": regular(ATTEMPT_ROOT / "reconciler_stderr.log") if (ATTEMPT_ROOT / "reconciler_stderr.log").is_file() else None,
                **FALSE_BOUNDARY,
            }
            snapshot = immutable_snapshot(terminal_state["context"]) if terminal_state["context"] is not None else None
            commit_terminal(failure, terminal_state, snapshot)
        raise
    finally:
        if terminal_state["visible"]:
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, signal.SIG_IGN)
        else:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def synthetic_self_test() -> int:
    import ast
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    popen_sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    spawn_source = ast.unparse(functions["spawn_owned"])
    actual_command = derive_r2_command({"sources": source_records()}) if R2.is_file() else []
    expected_flags = {
        "--preregistration", "--preregistration-sha", "--contract", "--contract-sha",
        "--old-static-source", "--old-static-sha", "--old-static-receipt", "--old-static-receipt-sha",
        "--old-static-log", "--old-static-log-sha", "--reconciliation-preregistration",
        "--reconciliation-preregistration-sha", "--repair-preregistration", "--repair-preregistration-sha",
        "--reconciliation-contract", "--reconciliation-contract-sha", "--reconciler-source",
        "--reconciler-sha", "--output",
    }
    r2_source = R2.read_text() if R2.is_file() else ""
    global ATTEMPT_ROOT, ATTEMPT_PREP
    saved_attempt_paths = (ATTEMPT_ROOT, ATTEMPT_PREP)
    saved_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    lifecycle_passed = False
    foreign_root_passed = False
    foreign_temp_passed = False
    member_replacement_passed = False
    terminal_faults_passed = False
    postcommit_priority_passed = False
    try:
        with tempfile.TemporaryDirectory(prefix="v506-compact-lifecycle-") as raw:
            ATTEMPT_ROOT = Path(raw) / "attempt"
            ATTEMPT_PREP = Path(raw) / "attempt.prep"
            stdout_stream, stderr_stream = commit_intent({"fixture": True},)
            close_durable(stdout_stream)
            close_durable(stderr_stream)
            fixture_snapshot = {"fixture": "stable"}
            fixture_state = {
                "context": {"fixture": True}, "visible": False,
                "committed_exact4": False, "record": None,
            }
            terminal = {"format": TERMINAL_FORMAT, "passed": True, "status": "fixture"}
            terminal_result = commit_terminal(
                terminal, fixture_state, fixture_snapshot, lambda _context: fixture_snapshot,
            )
            lifecycle_tree = exact_tree(ATTEMPT_ROOT)
            lifecycle_passed = (
                fixture_state["visible"] is True
                and fixture_state["committed_exact4"] is True
                and terminal_result == regular(ATTEMPT_ROOT / "terminal_receipt.json")
                and [path.name for path in sorted(ATTEMPT_ROOT.iterdir())] == [
                    "intent.json", "reconciler_stderr.log", "reconciler_stdout.log", "terminal_receipt.json",
                ]
                and lifecycle_tree["file_count"] == 4
                and not os.path.lexists(ATTEMPT_PREP)
            )

            # A foreign final attempt root is never overwritten or removed, and
            # the owned preparation directory is cleaned after NOREPLACE fails.
            ATTEMPT_ROOT = Path(raw) / "foreign-attempt"
            ATTEMPT_PREP = Path(raw) / "foreign-attempt.prep"
            ATTEMPT_ROOT.mkdir()
            foreign_marker = ATTEMPT_ROOT / "foreign.marker"
            foreign_marker.write_bytes(b"foreign\n")
            frozen_foreign = regular(foreign_marker)
            try:
                commit_intent({"fixture": "foreign-root"})
            except (FileExistsError, RuntimeError):
                pass
            else:
                raise RuntimeError("foreign attempt root accepted")
            foreign_root_passed = (
                regular(foreign_marker) == frozen_foreign
                and not os.path.lexists(ATTEMPT_PREP)
            )

            # Cleanup refuses all same-name replacements, including empty logs.
            member_replacement_passed = True
            for replaced_name in ("intent.json", "reconciler_stdout.log", "reconciler_stderr.log"):
                ATTEMPT_ROOT = Path(raw) / ("replacement-" + replaced_name.replace(".", "-"))
                ATTEMPT_PREP = ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + ".prep")
                replacement_payload = ("foreign replacement " + replaced_name + "\n").encode()

                def replace_member(prep, name=replaced_name, payload=replacement_payload):
                    path = prep / name
                    path.unlink()
                    descriptor = os.open(
                        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
                    )
                    try:
                        write_all(descriptor, payload)
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    fsync_directory(prep)
                    raise RuntimeError("injected member replacement")

                refused = False
                try:
                    commit_intent({"fixture": replaced_name}, replace_member)
                except RuntimeError as error:
                    refused = "owned member drift; refusing cleanup" in str(error)
                replacement_path = ATTEMPT_PREP / replaced_name
                member_replacement_passed = member_replacement_passed and (
                    refused
                    and ATTEMPT_PREP.is_dir()
                    and not os.path.lexists(ATTEMPT_ROOT)
                    and replacement_path.read_bytes() == replacement_payload
                )

            # A foreign JSON temporary blocks publication without touching it.
            foreign_json_root = Path(raw) / "foreign-json-temp"
            foreign_json_root.mkdir()
            foreign_json_target = foreign_json_root / "intent.json"
            foreign_json_temp = foreign_json_root / ".intent.json.v506.tmp"
            foreign_json_temp.write_bytes(b"foreign-temp\n")
            frozen_temp = regular(foreign_json_temp)
            try:
                write_noreplace(foreign_json_target, {"fixture": True})
            except FileExistsError:
                pass
            else:
                raise RuntimeError("foreign JSON temp accepted")
            foreign_temp_passed = regular(foreign_json_temp) == frozen_temp and not os.path.lexists(foreign_json_target)

            terminal_faults_passed = True
            fixture_snapshot = {"fixture": "stable"}
            for stage in ("after_link", "before_unlink", "before_dir_fsync", "after_dir_fsync"):
                ATTEMPT_ROOT = Path(raw) / ("terminal-fault-" + stage)
                ATTEMPT_PREP = Path(raw) / ("terminal-fault-" + stage + ".prep")
                ATTEMPT_ROOT.mkdir()
                write_noreplace(ATTEMPT_ROOT / "intent.json", {"fixture": stage})
                (ATTEMPT_ROOT / "reconciler_stdout.log").write_bytes(b"")
                (ATTEMPT_ROOT / "reconciler_stderr.log").write_bytes(b"")
                fsync_directory(ATTEMPT_ROOT)
                fault_state = {
                    "context": {"fixture": True}, "visible": False,
                    "committed_exact4": False, "record": None,
                }

                def inject(current_stage, wanted=stage):
                    if current_stage == wanted:
                        raise RuntimeError("injected " + wanted)

                try:
                    commit_terminal(
                        {"format": TERMINAL_FORMAT, "passed": True, "status": stage},
                        fault_state, fixture_snapshot, lambda _context: fixture_snapshot, inject,
                    )
                except RuntimeError as error:
                    if str(error) != "injected " + stage:
                        raise
                else:
                    raise RuntimeError("terminal fault did not propagate")
                terminal_path = ATTEMPT_ROOT / "terminal_receipt.json"
                frozen_terminal = regular(terminal_path)
                second_publish_rejected = False
                try:
                    commit_terminal(
                        {"format": TERMINAL_FORMAT, "passed": False, "status": "second"},
                        fault_state, fixture_snapshot, lambda _context: fixture_snapshot,
                    )
                except RuntimeError:
                    second_publish_rejected = True
                terminal_faults_passed = terminal_faults_passed and (
                    fault_state["visible"] is True
                    and fault_state["committed_exact4"] is False
                    and second_publish_rejected
                    and regular(terminal_path) == frozen_terminal
                    and [path.name for path in sorted(ATTEMPT_ROOT.iterdir())] == [
                        "intent.json", "reconciler_stderr.log", "reconciler_stdout.log", "terminal_receipt.json",
                    ]
                )

            # A completed exact4 terminal owns success priority across a pending signal.
            ATTEMPT_ROOT = Path(raw) / "postcommit-priority"
            ATTEMPT_PREP = Path(raw) / "postcommit-priority.prep"
            ATTEMPT_ROOT.mkdir()
            write_noreplace(ATTEMPT_ROOT / "intent.json", {"fixture": "postcommit"})
            (ATTEMPT_ROOT / "reconciler_stdout.log").write_bytes(b"")
            (ATTEMPT_ROOT / "reconciler_stderr.log").write_bytes(b"")
            priority_state = {
                "context": {"fixture": True}, "visible": False,
                "committed_exact4": False, "record": None,
            }
            commit_terminal(
                {"format": TERMINAL_FORMAT, "passed": True, "status": "postcommit"},
                priority_state, fixture_snapshot, lambda _context: fixture_snapshot,
            )
            priority_record = regular(ATTEMPT_ROOT / "terminal_receipt.json")
            signal.raise_signal(signal.SIGTERM)
            postcommit_priority_passed = (
                priority_state["committed_exact4"] is True
                and regular(ATTEMPT_ROOT / "terminal_receipt.json") == priority_record
            )
    finally:
        ATTEMPT_ROOT, ATTEMPT_PREP = saved_attempt_paths
        for signum, handler in saved_handlers.items():
            signal.signal(signum, handler)
    checks = {
        "single_popen_site": len(popen_sites) == 1 and popen_sites[0] in list(ast.walk(functions["spawn_owned"])),
        "spawn_has_signal_mask": "pthread_sigmask" in spawn_source,
        "spawn_sets_process_started_pgid": all(token in spawn_source for token in ("state['process']", "state['started']", "state['pgid']")),
        "terminal_noreplace": "write_noreplace" in ast.unparse(functions["commit_terminal"]),
        "no_direct_phase_a_or_training": "phase_a" not in derive_r2_command.__code__.co_names and "training" not in derive_r2_command.__code__.co_names,
        "command_has_unique_r2": ast.unparse(functions["derive_r2_command"]).count("str(R2)") == 2,
        "actual_command_exact_flags": bool(actual_command) and actual_command[:2] == [str(RLPY), str(R2)] and set(actual_command[2::2]) == expected_flags and actual_command[-1] == str(TRANSPARENT),
        "r2_parser_accepts_exact_flags": all(flag in r2_source for flag in expected_flags),
        "actual_f813_exact2": not F813.is_file() or tree_body(exact_f813_before()) == F813_EXACT2,
        "lifecycle_exact3_to_exact4": lifecycle_passed,
        "attempt_root_noreplace_foreign_preserved": foreign_root_passed,
        "json_temp_foreign_preserved": foreign_temp_passed,
        "attempt_member_replacements_refuse_cleanup": member_replacement_passed,
        "terminal_visible_uncommitted_faults": terminal_faults_passed,
        "terminal_postcommit_signal_priority": postcommit_priority_passed,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    print(json.dumps({"passed": True, "checks": checks, "checks_sha256": csha(checks)}, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    args = parse_args()
    if args.synthetic_self_test:
        raise RuntimeError("synthetic self-test takes no execution arguments")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
