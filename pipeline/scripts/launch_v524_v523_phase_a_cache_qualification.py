#!/usr/bin/env python3
"""One-shot v524 exact-source-roles-repaired RNG-isolated Phase-A launcher.

The launcher consumes the fresh v524 authority and exact-eight preregistration,
owns the fresh driver
process group, and restores both v218 services before publishing one terminal
receipt.  It grants no training, cache reuse, reward, or hidden-outcome access.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
SCRIPTS = ROOT / "pipeline/scripts"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"

SEED = 1660
AUTHORITY_CONTRACT = SCRIPTS / "v524_v523_phase_a_exact_source_roles_repair_execution_authority_contract.json"
AUTHORITY_MATERIALIZER = SCRIPTS / "materialize_v524_v523_phase_a_exact_source_roles_repair_execution_authority.py"
AUTHORITY_MATERIALIZER_SHA = "1b33ba41297a998379cceebe3fcb4085a59f017d46dc481fdc2d3859622767dd"
AUTHORITY_MATERIALIZER_BYTES = 35575
SELF = SCRIPTS / "launch_v524_v523_phase_a_cache_qualification.py"
DRIVER = SCRIPTS / "generate_v524_v523_phase_a_cache_qualification.py"
DRIVER_SHA = "c9f7c836781bb8311a37024068be6e5d080a5b8e56c44fffc2cef85002447e64"
DRIVER_BYTES = 48394
WORKER = SCRIPTS / "generate_v524_v523_phase_a_cache_qualification_worker.py"
WORKER_SHA = "51653b8cebea6fa5f75ccb5a3f3ffc4d3dd4bfc6436fac3c49a494102556252a"
WORKER_BYTES = 40272
AUDITOR = SCRIPTS / "audit_v524_v523_phase_a_rng_isolated_qualification.py"
AUDITOR_SHA = "fce5b2bfa906183f49c1fe96e4bae88bddf8efb891917a56e2bfe0a7804b5078"
AUDITOR_BYTES = 48260
RNG_PROXY = SCRIPTS / "v520_v519_rng_isolated_v169_runtime_proxy.py"
RNG_PROXY_SHA = "2b8ab0e2aec4a8de5d4bf8cffbb3525bbcd69b9b03270012a4a48f69f430b51b"
RNG_PROXY_BYTES = 8926
PACKAGE_MANIFEST = SCRIPTS / "v509_v508_wam_pipeline_package_tree_manifest.json"
PACKAGE_MANIFEST_SHA = "05c9da342cc6ee32d855e07f08a64c31f7c24ec9da8982fe8f3880e70027f623"
PACKAGE_MANIFEST_BYTES = 72190
FAILURE_FORENSIC = SCRIPTS / "v524_v523_phase_a_exact_source_roles_failure_forensic.json"
FAILURE_FORENSIC_SHA = "bda89f4244031c8b3719868a78477a5251ac864e399df0e8b47a619c8125d9f0"
FAILURE_FORENSIC_BYTES = 5650
AUTHORITY_ROOT = J / "v524_v523_phase_a_exact_source_roles_repair_execution_authority_seed1660_20260826"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")
AUTHORITY_RECEIPT = AUTHORITY_ROOT / "authority_receipt.json"
ATTEMPT_ROOT = J / "v524_v523_phase_a_cache_qualification_attempt_seed1660_20260826"
ATTEMPT_PREP = ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + ".attempt-prep")
QUALIFICATION_ROOT = Path("/root/v524_v523_phase_a_cache_qualification_seed1660_20260826")
QUALIFICATION_PREP = QUALIFICATION_ROOT.with_name(QUALIFICATION_ROOT.name + ".attempt-prep")

PREREG = J / "v524_v523_phase_a_cache_qualification_prereg_seed1660_20260826/preregistration.json"
PREREG_SHA = "133c13ffc994671f2309d98ca294f47e0b838e291156d1523f211dc9f3c60da7"
PREREG_BYTES = 57506
PHASE_A_CONTRACT = SCRIPTS / "v485_v482_v169_cache_determinism_scope_repair_contract.json"
PHASE_A_CONTRACT_SHA = "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
PHASE_A_CONTRACT_BYTES = 43960
FROZEN_LAUNCHER = SCRIPTS / "launch_v485_v169_cache_qualification.sh"
FROZEN_LAUNCHER_SHA = "f596a0411c9b0db03812440ab1ece036b27489e8f6bd978a443d1a2a9ae8587e"
FROZEN_LAUNCHER_BYTES = 23827
RESTART_SOURCE = SCRIPTS / "restart_v218_services.sh"
RESTART_SHA = "e54980c82266af22668a46cc9ee80b39ece66af53cc6d64d44621808807d51af"
RESTART_BYTES = 2794

V506_AUTHORITY_ROOT = J / "v506_v505_v503_compact_reconciliation_execution_authority_seed1647_20260825"
V506_AUTHORITY_RECEIPT = V506_AUTHORITY_ROOT / "authority_receipt.json"
V506_AUTHORITY_SHA = "33b72c430efbc0c96d21445b3f14ae31eef29f9a35566ced6cb8c9d582b33543"
V506_AUTHORITY_BYTES = 37311
V506_ATTEMPT = J / "v506_v505_v503_compact_single_reconciler_attempt_seed1647_20260825"
V506_TERMINAL = V506_ATTEMPT / "terminal_receipt.json"
V506_TERMINAL_SHA = "3c213ef203b8ee86ebbd83b9c1f5f89f886c4444a46fd4e2b6652973a8a7e931"
V506_TERMINAL_BYTES = 19783
V506_MATERIALIZATION_EVIDENCE = J / "v506_v505_v503_compact_reconciliation_execution_authority_materialization_evidence_seed1647_20260825"
V506_MATERIALIZATION_PROCESS = V506_MATERIALIZATION_EVIDENCE / "process_receipt.json"
V506_MATERIALIZATION_PROCESS_SHA = "2d68df738dfdc2d7b0a1a324aadeaf484f49314f02d06e83f93d454f8ad46129"
V506_MATERIALIZATION_PROCESS_BYTES = 18469
V507_AUTHORITY_ROOT = J / "v507_v506_phase_a_cache_qualification_execution_authority_seed1648_20260825"
V507_AUTHORITY_PREP = V507_AUTHORITY_ROOT.with_name(V507_AUTHORITY_ROOT.name + ".registration-prep")
V507_ATTEMPT_ROOT = J / "v507_v506_phase_a_cache_qualification_attempt_seed1648_20260825"
V507_ATTEMPT_PREP = V507_ATTEMPT_ROOT.with_name(V507_ATTEMPT_ROOT.name + ".attempt-prep")
V507_FAILURE_EVIDENCE = J / "v507_v506_phase_a_cache_qualification_execution_authority_materialization_evidence_seed1648_20260825"
V507_FAILURE_PROCESS = V507_FAILURE_EVIDENCE / "process_receipt.json"
V507_FAILURE_PROCESS_SHA = "6f707aacf01a428434300c8a914d87a77bbcc7d2e852004f1e60e1dda15bb2fd"
V507_FAILURE_PROCESS_BYTES = 2733
V507_FAILURE_STDERR = V507_FAILURE_EVIDENCE / "materializer_stderr.log"
V507_FAILURE_STDERR_SHA = "6f63f47fbdb0928292b119fb8e58d435dd0ca90d68d7bf777a2a66f812bc86c2"
V507_FAILURE_STDERR_BYTES = 1145
V507_FAILURE_STDOUT = V507_FAILURE_EVIDENCE / "materializer_stdout.log"
V507_FAILURE_FORENSIC = SCRIPTS / "v508_v507_phase_a_authority_prep_snapshot_failure_forensic.json"
V507_FAILURE_FORENSIC_SHA = "9ac9e9cd0c9dd8d39ad790194c74cc77d1f1ba94a30a18ae87f5e9e589db1e9b"
V507_FAILURE_FORENSIC_BYTES = 7821
V507_FAILURE_TREE = {
    "file_count": 6,
    "logical_file_bytes": 73672,
    "sha256sum_lines_digest_sha256": "89ecc2f7c2931ee32cebc0f9426cf4509b616889ae598b335650fb44496ddc19",
    "canonical_json_triples_digest_sha256": "3957eb0be4541b3d9da1f261bee533c29b616c181b242e0a32ad15af650bc89b",
}
TRANSPARENT = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
TRANSPARENT_SHA = "7740cabe53661da4832f2624642af01dac922e05aa6a62c6ef7882e8cb56a477"
TRANSPARENT_BYTES = 61920
F813 = TRANSPARENT.parent / "preregistration.json"
F813_SHA = "f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8"
F813_BYTES = 21296
F813_LOG = TRANSPARENT.parent / "immutable_evidence/v485_static_b73.log"
F813_LOG_SHA = "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b"
F813_LOG_BYTES = 6475
F813_TREE = {
    "file_count": 3,
    "logical_file_bytes": 89691,
    "sha256sum_lines_digest_sha256": "8628f5659dacbe3ede05699866ace077c4158e71b4c1020dcfadaf84d9d038b6",
    "canonical_json_triples_digest_sha256": "2c307584b6d902252365dd39b26dc6f59f38ac3b143034dc2d63529c83e9edaf",
}

RLPY = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
RLPY_RESOLVED = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
RLPY_RESOLVED_SHA = "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788"
RLPY_RESOLVED_BYTES = 25555040

AUTHORITY_FORMAT = "strict-track2-v524-v523-phase-a-exact-source-roles-repair-execution-authority-v1"
AUTHORITY_STATUS = "authorized_exact_one_external_v524_phase_a_cache_qualification_attempt"
CONTRACT_FORMAT = "strict-track2-v524-v523-phase-a-exact-source-roles-repair-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
INTENT_FORMAT = "strict-track2-v524-v523-phase-a-cache-qualification-attempt-intent-v1"
TERMINAL_FORMAT = "strict-track2-v524-v523-phase-a-cache-qualification-attempt-terminal-v1"
SOURCE_ORDER = [
    "authority_design_contract",
    "authority_materializer",
    "fresh_phase_a_launcher",
    "fresh_phase_a_driver",
    "fresh_phase_a_worker",
    "fresh_rng_proxy",
    "fresh_phase_a_independent_auditor",
    "fresh_phase_a_preregistration",
    "v524_v523_phase_a_failure_forensic",
    "v523_authority_design_contract",
    "v523_authority_materializer",
    "v523_authority_receipt",
    "v523_phase_a_launcher",
    "v523_phase_a_driver",
    "v523_phase_a_worker",
    "v523_phase_a_independent_auditor",
    "v523_attempt_intent",
    "v523_attempt_stdout",
    "v523_attempt_stderr",
    "v523_attempt_terminal",
    "phase_a_design_contract",
    "phase_a_output_materializer",
    "cache_scope_helper",
    "phase_a_static_auditor",
    "phase_a_static_audit",
    "restart_v218_source",
    "frozen_v485_launcher",
]
AUTHORIZATION = {
    "phase_a_cache_qualification_launcher_authorized": True,
    "launcher_invocations_authorized": 1,
    "launcher_invocations_consumed": 0,
    "nested_phase_a_driver_invocations_authorized": 1,
    "nested_phase_a_driver_only_via_launcher": True,
    "direct_phase_a_driver_authorized": False,
    "phase_a_worker_invocations_authorized": 2,
    "rng_proxy_required_for_every_runtime_delegate": True,
    "runtime_delegate_calls_authorized": 2000,
    "runtime_delegate_calls_consumed": 0,
    "worker_environment_exact": {"PYTHONPATH": str(ROOT / "pipeline"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    "service_environment_overrides_authorized": False,
    "retry_authorized": False,
    "cache_reuse_authorized": False,
    "training_authorized": False,
    "submission_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME_PRE = {
    "execution_authority_materialized": True,
    "phase_a_launcher_executed": False,
    "phase_a_driver_executed": False,
    "phase_a_worker_invocations": 0,
    "rng_proxy_delegate_invocations": 0,
    "qualification_output_created": False,
    "cache_reused": False,
    "training_launched": False,
    "reward_read": False,
    "dev_hidden_final_outcome_read": False,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True,
    "phase_a_launcher_invocations": 0,
    "phase_a_driver_invocations": 0,
    "phase_a_worker_invocations": 0,
    "rng_proxy_delegate_invocations": 0,
    "training_invocations": 0,
    "reward_reads": 0,
    "dev_hidden_final_outcome_reads": 0,
}
PER_CALL_RNG_CONTRACT = {
    "format": "strict-track2-v520-per-call-rng-isolation-evidence-v1",
    "expected_branches": ["A", "B"],
    "calls_per_branch": 1000,
    "total_runtime_delegate_calls": 2000,
    "durable_process_log_markers_per_call": 2,
    "durable_process_log_markers_per_branch": 2000,
    "total_durable_process_log_markers": 4000,
    "process_log_marker_order_per_call": ["started", "completed"],
    "required_rng_stage_keys": ["entry_external", "inside_before_delegate", "internal_after_delegate", "exit_restored"],
    "delegate_invocations_started_per_call": 1,
    "delegate_invocations_completed_per_successful_call": 1,
    "fork_rng_devices_exact_all_cuda_indices": True,
    "python_all_four_stages_equal": True,
    "numpy_all_four_stages_equal": True,
    "torch_cpu_exit_restored": True,
    "torch_cuda_exit_restored": True,
    "torch_internal_change_allowed": True,
    "raw_warning_order_preserved": True,
    "output_schema_recorded_per_call": True,
    "canonical_event_digest_recorded_per_call": True,
    "sample0_evidence_is_ancestry_not_runtime_substitute": True,
}
CHECK_KEYS = sorted({"contract_schema","source_order","source_aliases","source_closure_current","active_paths_exact","fixed_sources_exact","prereg_exact8","worker_env_exact2","per_call_contract_exact21","driver_per_call_exact21","worker_per_call_exact21","auditor_per_call_exact21","old_authority_exact1","old_authority_schema","old_authority_sources_current","old_authority_expected_current","v523_attempt_exact4","v523_terminal_failed_no_retry","v523_partition_exact","v523_root_cause_exact_ordered8","v523_forensic_exact","v523_cleanup_closed","v523_services_restored","v523_qualification_absent","fresh_roots_absent","authorization_boundary","execution_boundary","runtime_observation","input_prepost_equal","gpu_empty","no_live_phase_a_process","no_pending_values","publish_noreplace_exact1"})
CONTRACT_TOP_KEYS = {"format","status","seed","lineage","source_role_order","source_aliases","source_closure","source_closure_sha256","authority_materializer_source","phase_a_launcher_source","phase_a_driver_source","phase_a_worker_source","rng_proxy_source","phase_a_independent_auditor_source","phase_a_preregistration_source","qualification_output_root","worker_environment_exact","per_call_rng_evidence_contract","v523_authority_registration_tree","v523_authority_receipt","v523_authority_semantics","v523_attempt_tree","v523_attempt_intent","v523_attempt_stdout","v523_attempt_stderr","v523_attempt_terminal","v523_phase_a_failure_forensic","v523_phase_a_execution_partition","v523_exact_source_roles_failure_root_cause","historical_absences","current_absences_after_authority","authority_receipt_contract","authorization","runtime_observation","execution_boundary"}
SOURCE_ALIASES = {"authority_design_contract":"authority_design_contract", **{role:role+"_source" for role in SOURCE_ORDER[1:]}}
AUTH_TOP_KEYS = {"format","status","passed","source_closure","source_role_order","source_aliases","source_closure_sha256",*SOURCE_ALIASES.values(),"phase_a_launcher_source","phase_a_driver_source","phase_a_worker_source","rng_proxy_source","phase_a_independent_auditor_source","phase_a_preregistration_source","qualification_output_root","phase_a_attempt_root","worker_environment_exact","per_call_rng_evidence_contract","v519_evidence_tree","v519_process_receipt","v519_child_stdout","v519_output_record","v519_rng_isolation_evidence","v519_source_records","v517_candidate_tree","v517_candidate_receipt","v517_evidence_tree","v517_process_receipt","v517_normalized_current_view","old_phase_a_failure_forensics","historical_absences","required_absences","checks","check_keys","check_key_set_sha256","checks_sha256","input_pre_snapshot","input_post_snapshot","input_snapshots_exactly_equal","authorization","runtime_observation","execution_boundary","phase_a_cache_qualification_authorized","training_authorized","preregistration_sha256","contract_sha256"}

DRIVER_TIMEOUT_SECONDS = 9000
SERVICE_TIMEOUT_SECONDS = 120
TERM_GRACE_SECONDS = 15
_LIBC = ctypes.CDLL(None, use_errno=True) if os.name == "posix" else None
_RENAME_NOREPLACE = 1


class PendingSignal(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"pending signal {signum}")


def cjson(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def json_exact(left: object, right: object) -> bool:
    return cjson(left) == cjson(right)


def csha(value: object) -> str:
    return hashlib.sha256(cjson(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def regular(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"not regular: {path}")
    info = path.stat()
    record = {"path": str(path), "sha256": sha256(path), "logical_bytes": info.st_size}
    if expected_sha is not None and record["sha256"] != expected_sha:
        raise RuntimeError(f"sha mismatch: {path}")
    if expected_bytes is not None and record["logical_bytes"] != expected_bytes:
        raise RuntimeError(f"bytes mismatch: {path}")
    return record


def exact_tree(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"not regular directory: {root}")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"nonregular tree member: {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            rows.append([rel, sha256(path), path.stat().st_size])
    lines = b"".join(f"{row[1]}  {row[0]}\n".encode() for row in rows)
    return {
        "root": str(root),
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(cjson(rows)).hexdigest(),
    }


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(payload):
        count = os.write(fd, view[offset:])
        if count <= 0:
            raise OSError(errno.EIO, "short/zero write")
        offset += count


def rename_dir_noreplace(source: Path, target: Path) -> None:
    if _LIBC is None:
        raise RuntimeError("renameat2 requires the production POSIX host")
    result = _LIBC.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), _RENAME_NOREPLACE)
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(target))


def write_json_noreplace(path: Path, value: object, state: dict | None = None, fault_hook=None) -> dict:
    payload = cjson(value) + b"\n"
    temp = path.with_name(path.name + f".owned-{os.getpid()}-{time.time_ns()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    temp_stat = os.fstat(fd)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    temp_record = regular(temp, hashlib.sha256(payload).hexdigest(), len(payload))
    if temp_record["sha256"] != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("temporary JSON readback")
    linked = False
    try:
        os.link(temp, path, follow_symlinks=False)
        linked = True
        if state is not None:
            state["visible"] = True
        if fault_hook:
            fault_hook("after_terminal_link")
        target_stat = path.stat(follow_symlinks=False)
        if (target_stat.st_dev, target_stat.st_ino) != (temp_stat.st_dev, temp_stat.st_ino):
            raise RuntimeError("terminal inode drift")
        os.unlink(temp)
        if fault_hook:
            fault_hook("after_terminal_unlink")
        fsync_dir(path.parent)
        if fault_hook:
            fault_hook("after_terminal_dir_fsync")
        return regular(path, hashlib.sha256(payload).hexdigest(), len(payload))
    except BaseException:
        if not linked:
            owned_unlink(temp, (temp_stat.st_dev, temp_stat.st_ino), temp_record)
            fsync_dir(path.parent)
        raise


def owned_unlink(path: Path, identity: tuple[int, int], record: dict) -> None:
    if not os.path.lexists(path):
        return
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"refuse cleanup foreign member: {path}")
    if regular(path) != record:
        raise RuntimeError(f"refuse cleanup changed member: {path}")
    path.unlink()


def validate_attempt_owned(owned: dict, require_initial_logs: bool = False) -> dict:
    current = {}
    for name, value in owned.items():
        path = ATTEMPT_ROOT / name
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != value["identity"]:
            raise RuntimeError(f"attempt member identity drift: {name}")
        record = regular(path)
        if name == "intent.json" and record != value["record"]:
            raise RuntimeError("attempt intent drift")
        if require_initial_logs and name != "intent.json" and record["logical_bytes"] != 0:
            raise RuntimeError(f"attempt log not initially empty: {name}")
        current[name] = record
    return current


def validate_rlpy() -> dict:
    if not RLPY.is_symlink() or os.readlink(RLPY) != "/root/autodl-tmp/conda_envs/isaacsim51/bin/python":
        raise RuntimeError("RLPY lexical chain")
    middle = Path(os.readlink(RLPY))
    if not middle.is_symlink() or os.readlink(middle) != "python3.11":
        raise RuntimeError("RLPY middle chain")
    if RLPY.resolve(strict=True) != RLPY_RESOLVED:
        raise RuntimeError("RLPY resolved path")
    return regular(RLPY_RESOLVED, RLPY_RESOLVED_SHA, RLPY_RESOLVED_BYTES)


def source_alias(role: str) -> str:
    return "authority_design_contract" if role == "authority_design_contract" else role + "_source"


def validate_authority(args: argparse.Namespace) -> dict:
    if args.seed != SEED:
        raise RuntimeError("seed")
    contract_record = regular(args.authority_contract, args.authority_contract_sha, args.authority_contract_bytes)
    authority_record = regular(args.authority_receipt, args.authority_receipt_sha, args.authority_receipt_bytes)
    self_record = regular(args.launcher_source, args.launcher_sha, args.launcher_bytes)
    if args.authority_contract != AUTHORITY_CONTRACT or args.authority_receipt != AUTHORITY_RECEIPT or args.launcher_source != SELF:
        raise RuntimeError("canonical active paths")
    contract = json.loads(args.authority_contract.read_text())
    authority = json.loads(args.authority_receipt.read_text())
    authority_tree=exact_tree(AUTHORITY_ROOT)
    if authority_tree.get("file_count")!=1 or authority_tree.get("inventory")!=[["authority_receipt.json",authority_record["sha256"],authority_record["logical_bytes"]]] or os.path.lexists(AUTHORITY_PREP):raise RuntimeError("authority exact1/prep")
    receipt_schema=contract.get("authority_receipt_contract",{})
    if set(contract) != CONTRACT_TOP_KEYS or receipt_schema.get("top_keys")!=sorted(authority) or authority.get("check_keys") != CHECK_KEYS or set(authority.get("checks", {})) != set(CHECK_KEYS):
        raise RuntimeError("v520 schema counts")
    if not all(type(value) is bool and value is True for value in authority["checks"].values()):
        raise RuntimeError("v520 authority checks")
    if authority.get("check_key_set_sha256") != csha(CHECK_KEYS) or authority.get("checks_sha256") != csha(authority["checks"]):
        raise RuntimeError("v520 check digests")
    if contract.get("seed") != SEED:
        raise RuntimeError("contract seed")
    if contract.get("format") != CONTRACT_FORMAT or contract.get("status") != CONTRACT_STATUS:
        raise RuntimeError("contract terminal")
    if authority.get("format") != AUTHORITY_FORMAT or authority.get("status") != AUTHORITY_STATUS or authority.get("passed") is not True:
        raise RuntimeError("authority terminal")
    if (cjson(authority.get("authorization")) != cjson(AUTHORIZATION)
            or cjson(authority.get("runtime_observation")) != cjson(RUNTIME_PRE)
            or cjson(authority.get("execution_boundary")) != cjson(EXECUTION_BOUNDARY)
            or cjson(receipt_schema.get("authorization_exact")) != cjson(AUTHORIZATION)
            or cjson(receipt_schema.get("runtime_observation_exact")) != cjson(RUNTIME_PRE)
            or receipt_schema.get("format") != AUTHORITY_FORMAT
            or receipt_schema.get("status") != AUTHORITY_STATUS
            or receipt_schema.get("check_keys") != CHECK_KEYS
            or receipt_schema.get("check_key_set_sha256") != csha(CHECK_KEYS)
            or receipt_schema.get("checks_sha256") != csha(authority["checks"])):
        raise RuntimeError("authority boundary")
    if authority.get("phase_a_cache_qualification_authorized") is not True:
        raise RuntimeError("driver authority alias")
    if authority.get("training_authorized") is not False or authority.get("preregistration_sha256") != PREREG_SHA:
        raise RuntimeError("driver prereg alias")
    if authority.get("contract_sha256") != PHASE_A_CONTRACT_SHA or authority.get("qualification_output_root") != str(QUALIFICATION_ROOT):
        raise RuntimeError("driver contract/root alias")
    closure = authority.get("source_closure")
    order = authority.get("source_role_order")
    aliases = authority.get("source_aliases")
    if not isinstance(closure, dict) or order != SOURCE_ORDER or set(closure) != set(SOURCE_ORDER):
        raise RuntimeError("authority source order")
    expected_aliases = {role: source_alias(role) for role in SOURCE_ORDER}
    if aliases != expected_aliases:
        raise RuntimeError("authority source aliases")
    if authority.get("source_closure_sha256") != csha(closure):
        raise RuntimeError("authority source digest")
    contract_closure = contract.get("source_closure")
    contract_order = SOURCE_ORDER[1:]
    if (contract.get("source_role_order") != contract_order or set(contract_closure or {}) != set(contract_order)
            or contract.get("source_aliases") != {role: source_alias(role) for role in contract_order}
            or contract.get("source_closure_sha256") != csha(contract_closure)):
        raise RuntimeError("contract source closure")
    if any(contract_closure[role] != closure[role] for role in contract_order):
        raise RuntimeError("contract/authority source closure")
    for role in order:
        if authority.get(aliases[role]) != closure[role]:
            raise RuntimeError(f"authority source alias {role}")
        record = closure[role]
        if regular(Path(record["path"]), record["sha256"], record["logical_bytes"]) != record:
            raise RuntimeError(f"authority source current {role}")
    if closure["authority_design_contract"] != contract_record or closure["fresh_phase_a_launcher"] != self_record:
        raise RuntimeError("active source binding")
    required = {
        "authority_materializer": regular(AUTHORITY_MATERIALIZER, AUTHORITY_MATERIALIZER_SHA, AUTHORITY_MATERIALIZER_BYTES),
        "fresh_phase_a_driver": regular(DRIVER, DRIVER_SHA, DRIVER_BYTES),
        "fresh_phase_a_worker": regular(WORKER, WORKER_SHA, WORKER_BYTES),
        "fresh_rng_proxy": regular(RNG_PROXY, RNG_PROXY_SHA, RNG_PROXY_BYTES),
        "fresh_phase_a_preregistration": regular(PREREG, PREREG_SHA, PREREG_BYTES),
        "fresh_phase_a_independent_auditor": regular(AUDITOR, AUDITOR_SHA, AUDITOR_BYTES),
        "phase_a_static_audit": regular(TRANSPARENT, TRANSPARENT_SHA, TRANSPARENT_BYTES),
        "phase_a_design_contract": regular(PHASE_A_CONTRACT, PHASE_A_CONTRACT_SHA, PHASE_A_CONTRACT_BYTES),
        "frozen_v485_launcher": regular(FROZEN_LAUNCHER, FROZEN_LAUNCHER_SHA, FROZEN_LAUNCHER_BYTES),
        "restart_v218_source": regular(RESTART_SOURCE, RESTART_SHA, RESTART_BYTES),
    }
    for role, record in required.items():
        if closure[role] != record:
            raise RuntimeError(f"frozen binding {role}")
    lineage = {
        "name": "v520_v519_phase_a_rng_isolated_qualification",
        "fresh_authority_root": str(AUTHORITY_ROOT),
        "fresh_phase_a_attempt_root": str(ATTEMPT_ROOT),
        "fresh_qualification_output_root": str(QUALIFICATION_ROOT),
        "supersedes_old_phase_a_failures_no_retry": True,
        "sole_semantic_change": "per_runtime_delegate_rng_fork_restore_with_full_per_call_evidence",
    }
    if (cjson(contract.get("lineage")) != cjson(lineage) or contract.get("authority_materializer_source") != closure["authority_materializer"]
            or contract.get("phase_a_launcher_source") != self_record
            or contract.get("phase_a_driver_source") != closure["fresh_phase_a_driver"]
            or contract.get("phase_a_worker_source") != closure["fresh_phase_a_worker"]
            or contract.get("rng_proxy_source") != closure["fresh_rng_proxy"]
            or contract.get("phase_a_independent_auditor_source") != closure["fresh_phase_a_independent_auditor"]
            or contract.get("phase_a_preregistration_source") != closure["fresh_phase_a_preregistration"]
            or contract.get("qualification_output_root") != str(QUALIFICATION_ROOT)
            or contract.get("worker_environment_exact") != AUTHORIZATION["worker_environment_exact"]):
        raise RuntimeError("v520 contract active lineage")
    if (cjson(contract.get("authorization")) != cjson(AUTHORIZATION)
            or cjson(contract.get("runtime_observation")) != cjson(RUNTIME_PRE)
            or cjson(contract.get("execution_boundary")) != cjson(EXECUTION_BOUNDARY)):
        raise RuntimeError("v520 contract boundary")
    if authority.get("authority_design_contract") != contract_record:
        raise RuntimeError("contract alias")
    if (cjson(contract.get("per_call_rng_evidence_contract")) != cjson(PER_CALL_RNG_CONTRACT)
            or cjson(authority.get("per_call_rng_evidence_contract")) != cjson(PER_CALL_RNG_CONTRACT)
            or cjson(receipt_schema.get("per_call_rng_evidence_contract_exact")) != cjson(PER_CALL_RNG_CONTRACT)):
        raise RuntimeError("per-call RNG contract")
    v519_tree=authority.get("v519_evidence_tree")
    if v519_tree!=contract.get("v519_evidence_tree") or v519_tree.get("file_count")!=6 or v519_tree.get("logical_file_bytes")!=653273 or v519_tree.get("sha256sum_lines_digest_sha256")!="f386dec633821d8e85b7097eed5fcc6668c07958ee52aaab9327804286bfbd26" or v519_tree.get("canonical_json_triples_digest_sha256")!="e06c481df6865a157b6d34dab36fd99f867689a6eddad0309484f72885ab0538":raise RuntimeError("v519 exact6")
    for key,sha_value,byte_value in (("v519_process_receipt","06ce4ad84ad9bd3efdf35b429dbf5c0f1986c4d32ae495afcbc7a1e97060ed11",369884),("v519_child_stdout","6585fbb99e063cb86287ca109ce7c4595dc0d07be1330f914ab53d83be06963d",126129)):
        value=authority.get(key)
        if value!=contract.get(key) or value.get("sha256")!=sha_value or value.get("logical_bytes")!=byte_value or regular(Path(value["path"]),sha_value,byte_value)!=value:raise RuntimeError(key)
    if contract.get("v519_output_record",{}).get("sha256")!="119d47797f50b24b2050c91f99b1733b9f2c0ed20a7987f61d44ca42ef3dfce3" or authority.get("v519_output_record")!=contract.get("v519_output_record"):raise RuntimeError("v519 output")
    rng=contract.get("v519_rng_isolation_evidence",{})
    if authority.get("v519_rng_isolation_evidence")!=rng or rng.get("cuda_device_count")!=1 or rng.get("cuda_device_indices")!=[0] or any(rng.get(key) is not True for key in ("python_all_four_stages_equal","numpy_all_four_stages_equal","torch_cpu_exit_restored","torch_cuda_exit_restored")):raise RuntimeError("v519 RNG ancestry")
    for key in ("v517_candidate_tree","v517_evidence_tree"):
        if authority.get(key)!=contract.get(key):raise RuntimeError(key)
    for key in ("v517_candidate_receipt","v517_process_receipt"):
        value=authority.get(key)
        if value!=contract.get(key) or regular(Path(value["path"]),value["sha256"],value["logical_bytes"])!=value:raise RuntimeError(key)
    if authority.get("v517_normalized_current_view")!=contract.get("v517_normalized_current_view"):raise RuntimeError("v517 normalized current")
    failures=contract.get("old_phase_a_failure_forensics",{})
    if authority.get("old_phase_a_failure_forensics")!=failures or set(failures)!={"v509_import_failure_forensic","v509_warning_failure_forensic","v511_anchor_failure_forensic","v518_rng_failure_forensic"}:raise RuntimeError("failure ancestry")
    for role,value in failures.items():
        record = closure[role]
        if regular(Path(record["path"]), record["sha256"], record["logical_bytes"]) != record:
            raise RuntimeError(role + " record")
        if json.loads(Path(record["path"]).read_text()) != value:
            raise RuntimeError(role + " value")
    prereg = json.loads(PREREG.read_text())
    if prereg.get("qualification_output_root") != str(QUALIFICATION_ROOT):
        raise RuntimeError("v520 exact8 root")
    records = prereg.get("execution_source_records")
    if not isinstance(records, list) or len(records) != 8 or csha(records) != prereg["execution_sources_digest_sha256"]:
        raise RuntimeError("v520 exact8 records")
    for record in records:
        if regular(Path(record["path"]), record["sha256"], record["logical_bytes"]) != {k: record[k] for k in ("path", "sha256", "logical_bytes")}:
            raise RuntimeError("v520 exact8 current")
    if prereg["execution_sources"].get("phase_a_driver") != closure["fresh_phase_a_driver"] or prereg["execution_sources"].get("phase_a_process_worker") != closure["fresh_phase_a_worker"] or prereg["execution_sources"].get("phase_a_rng_isolation_proxy") != closure["fresh_rng_proxy"]:raise RuntimeError("fresh execution prereg")
    for record in authority.get("required_absences", {}).values():
        if record.get("absent") is not True or os.path.lexists(record.get("path", "")):raise RuntimeError("fresh required absence")
    validate_rlpy()
    return {"contract": contract, "authority": authority, "closure": closure, "preregistration": prereg,
            "contract_record": contract_record, "authority_record": authority_record, "self_record": self_record,
            "failure_forensics": failures}


def stable_snapshot(context: dict) -> dict:
    records = [context["contract_record"], context["authority_record"], context["self_record"]]
    records.extend(context["closure"][role] for role in SOURCE_ORDER)
    records.extend(context["authority"][key] for key in ("v519_process_receipt","v519_child_stdout","v517_candidate_receipt","v517_process_receipt"))
    records.extend(context["authority"]["old_phase_a_failure_forensics"].values())
    authority=context["authority"]
    trees={key:exact_tree(Path(authority[key]["root"])) for key in ("v519_evidence_tree","v517_candidate_tree","v517_evidence_tree")}
    return {"records":records,"records_sha256":csha(records),"ancestry_trees":trees,"v519_rng_isolation_evidence":authority["v519_rng_isolation_evidence"],"v517_normalized_current_view":authority["v517_normalized_current_view"]}


def load_v524_materializer():
    source = AUTHORITY_MATERIALIZER.read_bytes()
    if (AUTHORITY_MATERIALIZER.is_symlink() or not AUTHORITY_MATERIALIZER.is_file()
            or hashlib.sha256(source).hexdigest() != AUTHORITY_MATERIALIZER_SHA
            or len(source) != AUTHORITY_MATERIALIZER_BYTES):
        raise RuntimeError("v524 materializer source")
    spec = importlib.util.spec_from_file_location("v524_authority_materializer_consumer", AUTHORITY_MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(AUTHORITY_MATERIALIZER), "exec"), module.__dict__)
    return module


def validate_authority(args: argparse.Namespace) -> dict:
    if type(args.seed) is not int or args.seed != SEED:
        raise RuntimeError("seed")
    contract_record = regular(args.authority_contract, args.authority_contract_sha, args.authority_contract_bytes)
    authority_record = regular(args.authority_receipt, args.authority_receipt_sha, args.authority_receipt_bytes)
    self_record = regular(args.launcher_source, args.launcher_sha, args.launcher_bytes)
    if args.authority_contract != AUTHORITY_CONTRACT or args.authority_receipt != AUTHORITY_RECEIPT or args.launcher_source != SELF:
        raise RuntimeError("canonical active paths")
    materializer = load_v524_materializer()
    if (materializer.CONTRACT_PATH != AUTHORITY_CONTRACT or materializer.AUTHORITY_ROOT != AUTHORITY_ROOT
            or materializer.ATTEMPT_ROOT != ATTEMPT_ROOT or materializer.QUALIFICATION_ROOT != QUALIFICATION_ROOT
            or materializer.PREREG_PATH != PREREG or materializer.CONTRACT_SOURCE_ORDER != SOURCE_ORDER[1:]
            or materializer.AUTHORITY_SOURCE_ORDER != SOURCE_ORDER
            or materializer.CHECK_KEYS != CHECK_KEYS or materializer.CONTRACT_TOP_KEYS != CONTRACT_TOP_KEYS
            or materializer.CONTRACT_FORMAT != CONTRACT_FORMAT or materializer.CONTRACT_STATUS != CONTRACT_STATUS
            or materializer.OUTPUT_FORMAT != AUTHORITY_FORMAT or materializer.OUTPUT_STATUS != AUTHORITY_STATUS
            or materializer.AUTHORIZATION != AUTHORIZATION or materializer.RUNTIME != RUNTIME_PRE
            or materializer.EXECUTION_BOUNDARY != EXECUTION_BOUNDARY
            or not json_exact(materializer.PER_CALL_RNG_CONTRACT, PER_CALL_RNG_CONTRACT)):
        raise RuntimeError("materializer/launcher interface")
    contract = json.loads(args.authority_contract.read_text())
    authority = json.loads(args.authority_receipt.read_text())
    authority_tree = exact_tree(AUTHORITY_ROOT)
    if (authority_tree.get("file_count") != 1
            or authority_tree.get("inventory") != [["authority_receipt.json", authority_record["sha256"], authority_record["logical_bytes"]]]
            or os.path.lexists(AUTHORITY_PREP)):
        raise RuntimeError("authority exact1/prep")
    receipt_schema = contract.get("authority_receipt_contract", {})
    if (set(contract) != CONTRACT_TOP_KEYS or len(contract) != 35
            or sorted(authority) != receipt_schema.get("top_keys")
            or len(authority) != len(receipt_schema.get("top_keys", []))
            or authority.get("check_keys") != CHECK_KEYS or set(authority.get("checks", {})) != set(CHECK_KEYS)
            or not all(type(value) is bool and value is True for value in authority["checks"].values())
            or authority.get("check_key_set_sha256") != csha(CHECK_KEYS)
            or authority.get("checks_sha256") != csha(authority["checks"])
            or contract.get("seed") != SEED or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or authority.get("format") != AUTHORITY_FORMAT
            or authority.get("status") != AUTHORITY_STATUS or authority.get("passed") is not True):
        raise RuntimeError("v524 schema/terminal")
    receipt_schema_keys = {"top_keys", "check_keys", "source_role_order_exact", "source_aliases_exact",
                           "authorization_exact", "per_call_rng_evidence_contract_exact"}
    if (set(receipt_schema) != receipt_schema_keys
            or not json_exact(authority.get("authorization"), AUTHORIZATION)
            or not json_exact(authority.get("runtime_observation"), RUNTIME_PRE)
            or not json_exact(authority.get("execution_boundary"), EXECUTION_BOUNDARY)
            or not json_exact(receipt_schema.get("authorization_exact"), AUTHORIZATION)
            or not json_exact(authority.get("per_call_rng_evidence_contract"), PER_CALL_RNG_CONTRACT)
            or receipt_schema.get("check_keys") != CHECK_KEYS
            or receipt_schema.get("source_role_order_exact") != SOURCE_ORDER
            or not json_exact(receipt_schema.get("source_aliases_exact"), materializer.SOURCE_ALIASES)
            or not json_exact(receipt_schema.get("per_call_rng_evidence_contract_exact"), PER_CALL_RNG_CONTRACT)):
        raise RuntimeError("v524 authority boundary")
    authorization_count_keys = (
        "launcher_invocations_authorized", "launcher_invocations_consumed",
        "nested_phase_a_driver_invocations_authorized", "phase_a_worker_invocations_authorized",
        "runtime_delegate_calls_authorized", "runtime_delegate_calls_consumed")
    if (any(type(authority["authorization"].get(key)) is not int for key in authorization_count_keys)
            or any(type(receipt_schema["authorization_exact"].get(key)) is not int for key in authorization_count_keys)):
        raise RuntimeError("v524 authorization count types")
    closure = authority.get("source_closure")
    if (authority.get("source_role_order") != SOURCE_ORDER or set(closure or {}) != set(SOURCE_ORDER)
            or authority.get("source_aliases") != materializer.SOURCE_ALIASES
            or authority.get("source_closure_sha256") != csha(closure)):
        raise RuntimeError("v524 authority source closure")
    contract_closure = contract.get("source_closure")
    if (contract.get("source_role_order") != SOURCE_ORDER[1:] or set(contract_closure or {}) != set(SOURCE_ORDER[1:])
            or contract.get("source_aliases") != {role: materializer.SOURCE_ALIASES[role] for role in SOURCE_ORDER[1:]}
            or contract.get("source_closure_sha256") != csha(contract_closure)
            or any(contract_closure[role] != closure[role] for role in SOURCE_ORDER[1:])):
        raise RuntimeError("v524 contract source closure")
    for role in SOURCE_ORDER:
        record = closure[role]
        if authority.get(materializer.SOURCE_ALIASES[role]) != record:
            raise RuntimeError("v524 source alias " + role)
        if regular(Path(record["path"]), record["sha256"], record["logical_bytes"]) != record:
            raise RuntimeError("v524 source current " + role)
    expected_active = {
        "authority_design_contract": contract_record,
        "authority_materializer": regular(AUTHORITY_MATERIALIZER, AUTHORITY_MATERIALIZER_SHA, AUTHORITY_MATERIALIZER_BYTES),
        "fresh_phase_a_launcher": self_record,
        "fresh_phase_a_driver": regular(DRIVER, DRIVER_SHA, DRIVER_BYTES),
        "fresh_phase_a_worker": regular(WORKER, WORKER_SHA, WORKER_BYTES),
        "fresh_rng_proxy": regular(RNG_PROXY, RNG_PROXY_SHA, RNG_PROXY_BYTES),
        "fresh_phase_a_independent_auditor": regular(AUDITOR, AUDITOR_SHA, AUDITOR_BYTES),
        "fresh_phase_a_preregistration": regular(PREREG, PREREG_SHA, PREREG_BYTES),
    }
    if any(closure[role] != record for role, record in expected_active.items()):
        raise RuntimeError("v524 active source binding")
    expected_lineage = {
        "name": "v524_v523_phase_a_exact_source_roles_repair",
        "fresh_authority_root": str(AUTHORITY_ROOT),
        "fresh_phase_a_attempt_root": str(ATTEMPT_ROOT),
        "fresh_qualification_output_root": str(QUALIFICATION_ROOT),
        "supersedes_v523_failed_attempt_no_retry": True,
        "old_v523_authority_expected_current_and_consumed_once": True,
        "sole_semantic_change": "formal_exact_roles_equal_execution_source_records_ordered8",
    }
    if (not json_exact(contract.get("lineage"), expected_lineage)
            or not json_exact(contract.get("authorization"), AUTHORIZATION)
            or not json_exact(contract.get("runtime_observation"), RUNTIME_PRE)
            or not json_exact(contract.get("execution_boundary"), EXECUTION_BOUNDARY)
            or contract.get("phase_a_launcher_source") != self_record
            or contract.get("phase_a_driver_source") != closure["fresh_phase_a_driver"]
            or contract.get("phase_a_worker_source") != closure["fresh_phase_a_worker"]
            or contract.get("rng_proxy_source") != closure["fresh_rng_proxy"]
            or contract.get("phase_a_independent_auditor_source") != closure["fresh_phase_a_independent_auditor"]
            or contract.get("phase_a_preregistration_source") != closure["fresh_phase_a_preregistration"]
            or contract.get("qualification_output_root") != str(QUALIFICATION_ROOT)
            or not json_exact(contract.get("worker_environment_exact"), AUTHORIZATION["worker_environment_exact"])
            or not json_exact(contract.get("per_call_rng_evidence_contract"), PER_CALL_RNG_CONTRACT)):
        raise RuntimeError("v524 contract active lineage")
    old = materializer.validate_old_state()
    exact_anchors = {
        "v523_authority_registration_tree": old["authority_tree"],
        "v523_authority_receipt": old["authority_receipt"],
        "v523_authority_semantics": old["authority_semantics"],
        "v523_attempt_tree": old["attempt_tree"],
        "v523_attempt_intent": old["attempt_intent"],
        "v523_attempt_stdout": old["attempt_stdout"],
        "v523_attempt_stderr": old["attempt_stderr"],
        "v523_attempt_terminal": old["attempt_terminal"],
        "v523_phase_a_failure_forensic": old["failure_forensic"],
        "v523_phase_a_execution_partition": old["execution_partition"],
        "v523_exact_source_roles_failure_root_cause": old["root_cause"],
    }
    for key, expected in exact_anchors.items():
        if not json_exact(contract.get(key), expected) or not json_exact(authority.get(key), expected):
            raise RuntimeError("v524 old-state anchor " + key)
    partition_count_keys = (
        "launcher_invocations", "nested_phase_a_driver_invocations",
        "direct_phase_a_driver_invocations", "phase_a_worker_invocations",
        "rng_proxy_delegate_invocations", "training_invocations", "retry_invocations")
    if any(type(authority["v523_phase_a_execution_partition"].get(key)) is not int for key in partition_count_keys):
        raise RuntimeError("v524 old partition count types")
    prereg = materializer.validate_prereg(closure)
    for path in (DRIVER, WORKER, AUDITOR):
        if not json_exact(materializer.literal_assignment(path, "PER_CALL_RNG_CONTRACT"), PER_CALL_RNG_CONTRACT):
            raise RuntimeError("v524 per-call exact21 " + str(path))
    if authority.get("phase_a_cache_qualification_authorized") is not True or authority.get("training_authorized") is not False:
        raise RuntimeError("driver authorization aliases")
    if (authority.get("preregistration_sha256") != PREREG_SHA
            or authority.get("contract_sha256") != PHASE_A_CONTRACT_SHA
            or authority.get("qualification_output_root") != str(QUALIFICATION_ROOT)):
        raise RuntimeError("driver formal aliases")
    historical, current = materializer.absence_paths()
    contract_historical = {key: {"path": str(path)} for key, path in historical.items()}
    contract_current = {key: {"path": str(path)} for key, path in current.items()}
    receipt_historical = {key: {"path": str(path), "absent": True} for key, path in historical.items()}
    receipt_current = {key: {"path": str(path), "absent": True} for key, path in current.items()}
    if contract.get("historical_absences") != contract_historical:
        raise RuntimeError("v524 historical absence record")
    if contract.get("current_absences_after_authority") != contract_current:
        raise RuntimeError("v524 current absence record")
    if authority.get("historical_absences") != receipt_historical:
        raise RuntimeError("v524 historical absence ancestry")
    if authority.get("required_absences") != receipt_current:
        raise RuntimeError("v524 required absences")
    for row in authority.get("required_absences", {}).values():
        if row.get("absent") is not True or os.path.lexists(row.get("path", "")):
            raise RuntimeError("fresh required absence")
    if authority.get("input_snapshots_exactly_equal") is not True or authority.get("input_pre_snapshot") != authority.get("input_post_snapshot"):
        raise RuntimeError("v524 input snapshots")
    validate_rlpy()
    return {"contract": contract, "authority": authority, "closure": closure, "preregistration": prereg,
            "contract_record": contract_record, "authority_record": authority_record, "self_record": self_record,
            "failure_forensics": {"v523_exact_source_roles_failure": closure["v524_v523_phase_a_failure_forensic"]}}


def stable_snapshot(context: dict) -> dict:
    authority = context["authority"]
    records = [context["contract_record"], context["authority_record"], context["self_record"]]
    records.extend(context["closure"][role] for role in SOURCE_ORDER)
    tree_keys = ("v523_authority_registration_tree", "v523_attempt_tree")
    trees = {key: exact_tree(Path(authority[key]["root"])) for key in tree_keys}
    if any(trees[key] != authority[key] for key in tree_keys):
        raise RuntimeError("v524 ancestry tree drift")
    return {"records": records, "records_sha256": csha(records), "ancestry_trees": trees,
            "v523_authority_semantics": authority["v523_authority_semantics"],
            "v523_phase_a_failure_forensic": authority["v523_phase_a_failure_forensic"],
            "v523_exact_source_roles_failure_root_cause": authority["v523_exact_source_roles_failure_root_cause"]}


def health_codes() -> dict:
    result = {}
    for key, url in (("8005_v1_health", "http://127.0.0.1:8005/v1/health"), ("18084_health", "http://127.0.0.1:18084/health")):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                body = response.read()
                result[key] = {"returncode": 0, "http_code": response.status,
                               "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body)}
        except Exception:
            result[key] = {"returncode": -1, "http_code": 0, "body_sha256": hashlib.sha256(b"").hexdigest(), "body_bytes": 0}
    return result


def services_healthy(codes: dict) -> bool:
    return all(row["returncode"] == 0 and row["http_code"] == 200 and row["body_bytes"] > 0 for row in codes.values())


def services_stopped(codes: dict) -> bool:
    return all(row["returncode"] != 0 or row["http_code"] != 200 for row in codes.values())


def gpu_pids() -> list[int]:
    blocked = {signal.SIGINT, signal.SIGTERM}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    proc = None
    try:
        proc = subprocess.Popen(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, text=True)
        pgid = proc.pid
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        raise
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        stdout, _stderr = proc.communicate(timeout=10)
    except BaseException:
        if proc.poll() is None:
            try: os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError: pass
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError("GPU query")
    try: os.killpg(pgid, 0)
    except ProcessLookupError: pass
    else: raise RuntimeError("GPU query group not empty")
    return [int(line.strip()) for line in stdout.splitlines() if line.strip().isdigit()]


def run_service(action: str) -> dict:
    blocked = {signal.SIGINT, signal.SIGTERM}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    proc = None
    try:
        proc = subprocess.Popen(["/bin/bash", str(RESTART_SOURCE), action], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        pgid = proc.pid
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        raise
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        stdout, stderr = proc.communicate(timeout=SERVICE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        raise RuntimeError(f"v218 {action} timeout")
    except BaseException:
        if proc.poll() is None:
            try: os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try: os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=TERM_GRACE_SECONDS)
        else:
            proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"v218 {action} rc={proc.returncode} stderr_sha={hashlib.sha256(stderr).hexdigest()}")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        group_empty = True
    else:
        group_empty = False
    if not group_empty:
        raise RuntimeError(f"v218 {action} group not empty")
    return {"action": action, "returncode": proc.returncode, "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stdout_bytes": len(stdout), "stderr_sha256": hashlib.sha256(stderr).hexdigest(), "stderr_bytes": len(stderr),
            "pid": proc.pid, "pgid": pgid, "reaped": True, "group_empty": True}


def wait_health(expected_healthy: bool) -> dict:
    deadline = time.monotonic() + SERVICE_TIMEOUT_SECONDS
    last = health_codes()
    while time.monotonic() < deadline:
        last = health_codes()
        if (services_healthy(last) if expected_healthy else services_stopped(last)):
            return {"expected_healthy": expected_healthy, "observed": last, "passed": True}
        time.sleep(1)
    return {"expected_healthy": expected_healthy, "observed": last, "passed": False}


def derive_driver_command(context: dict) -> list[str]:
    closure = context["closure"]
    return [str(RLPY), closure["fresh_phase_a_driver"]["path"],
            "--preregistration", closure["fresh_phase_a_preregistration"]["path"], "--preregistration-sha", closure["fresh_phase_a_preregistration"]["sha256"],
            "--contract", str(PHASE_A_CONTRACT),
            "--authority-receipt", str(AUTHORITY_RECEIPT), "--authority-receipt-sha", context["authority_record"]["sha256"],
            "--scope-source", closure["cache_scope_helper"]["path"], "--scope-sha", closure["cache_scope_helper"]["sha256"],
            "--rng-proxy-source", closure["fresh_rng_proxy"]["path"], "--rng-proxy-sha", closure["fresh_rng_proxy"]["sha256"], "--rng-proxy-bytes", str(closure["fresh_rng_proxy"]["logical_bytes"]),
            "--worker-source", closure["fresh_phase_a_worker"]["path"], "--worker-sha", closure["fresh_phase_a_worker"]["sha256"],
            "--auditor-source", closure["fresh_phase_a_independent_auditor"]["path"], "--auditor-sha", closure["fresh_phase_a_independent_auditor"]["sha256"],
            "--materializer-source", closure["phase_a_output_materializer"]["path"], "--materializer-sha", closure["phase_a_output_materializer"]["sha256"],
            "--static-auditor-source", closure["phase_a_static_auditor"]["path"], "--static-auditor-sha", closure["phase_a_static_auditor"]["sha256"],
            "--launcher-source", str(FROZEN_LAUNCHER), "--launcher-sha", FROZEN_LAUNCHER_SHA,
            "--driver-source", closure["fresh_phase_a_driver"]["path"], "--driver-sha", closure["fresh_phase_a_driver"]["sha256"], "--driver-bytes", str(closure["fresh_phase_a_driver"]["logical_bytes"]),
            "--package-manifest", str(PACKAGE_MANIFEST), "--package-manifest-sha", PACKAGE_MANIFEST_SHA, "--package-manifest-bytes", str(PACKAGE_MANIFEST_BYTES)]


def commit_intent(intent: dict, fault_hook=None) -> tuple[object, object, dict]:
    if os.path.lexists(ATTEMPT_ROOT) or os.path.lexists(ATTEMPT_PREP):
        raise FileExistsError("fresh attempt state")
    ATTEMPT_PREP.mkdir(parents=False)
    prep_stat = ATTEMPT_PREP.stat()
    owned = {}
    try:
        intent_path = ATTEMPT_PREP / "intent.json"
        intent_record = write_json_noreplace(intent_path, intent)
        intent_info = intent_path.stat(follow_symlinks=False)
        owned["intent.json"] = {"identity": (intent_info.st_dev, intent_info.st_ino), "record": intent_record, "stream": None}
        for name in ("phase_a_stdout.log", "phase_a_stderr.log"):
            path = ATTEMPT_PREP / name
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            stream = os.fdopen(fd, "w+b", buffering=0)
            info = os.fstat(fd)
            owned[name] = {"identity": (info.st_dev, info.st_ino), "record": regular(path), "stream": stream}
        fsync_dir(ATTEMPT_PREP)
        if fault_hook: fault_hook("before_attempt_promote")
        rename_dir_noreplace(ATTEMPT_PREP, ATTEMPT_ROOT)
        fsync_dir(ATTEMPT_ROOT.parent)
        root_stat = ATTEMPT_ROOT.stat()
        if (root_stat.st_dev, root_stat.st_ino) != (prep_stat.st_dev, prep_stat.st_ino):
            raise RuntimeError("attempt promoted inode")
        tree = exact_tree(ATTEMPT_ROOT)
        if [row[0] for row in tree["inventory"]] != ["intent.json", "phase_a_stderr.log", "phase_a_stdout.log"]:
            raise RuntimeError("attempt exact3 intent state")
        for name, value in owned.items():
            final_path = ATTEMPT_ROOT / name
            info = final_path.stat(follow_symlinks=False)
            value["identity"] = (info.st_dev, info.st_ino)
            value["record"] = regular(final_path)
        validate_attempt_owned(owned, require_initial_logs=True)
        return owned["phase_a_stdout.log"]["stream"], owned["phase_a_stderr.log"]["stream"], owned
    except BaseException:
        for value in owned.values():
            try:
                if value["stream"] is not None: value["stream"].close()
            except Exception: pass
        if ATTEMPT_PREP.exists() and not ATTEMPT_PREP.is_symlink():
            current = ATTEMPT_PREP.stat()
            if (current.st_dev, current.st_ino) == (prep_stat.st_dev, prep_stat.st_ino):
                for name, value in owned.items():
                    owned_unlink(ATTEMPT_PREP / name, value["identity"], value["record"])
                ATTEMPT_PREP.rmdir(); fsync_dir(ATTEMPT_PREP.parent)
        raise


def validate_durable_process_markers(path: Path, role: str) -> dict:
    rows = []
    for line in path.read_text(errors="strict").splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if row.get("event") in ("rng_proxy_call_started", "rng_proxy_call_completed"):
                rows.append(row)
    if len(rows) != PER_CALL_RNG_CONTRACT["durable_process_log_markers_per_branch"]:
        raise RuntimeError("launcher durable marker count " + role)
    expected_samples = list(range(1000)) if role == "A" else list(reversed(range(1000)))
    for ordinal, sample_id in enumerate(expected_samples):
        started, completed = rows[2 * ordinal:2 * ordinal + 2]
        if ([started.get("event").removeprefix("rng_proxy_call_"), completed.get("event").removeprefix("rng_proxy_call_")]
                != PER_CALL_RNG_CONTRACT["process_log_marker_order_per_call"]
                or type(started.get("ordinal")) is not int or started.get("ordinal") != ordinal
                or type(completed.get("ordinal")) is not int or completed.get("ordinal") != ordinal
                or type(started.get("sample_id")) is not int or started.get("sample_id") != sample_id
                or type(completed.get("sample_id")) is not int or completed.get("sample_id") != sample_id
                or type(started.get("started")) is not int or started.get("started") != 1
                or type(started.get("completed")) is not int or started.get("completed") != 0
                or type(completed.get("started")) is not int or completed.get("started") != 1
                or type(completed.get("completed")) is not int or completed.get("completed") != 1):
            raise RuntimeError("launcher durable marker order " + role)
    return {"role": role, "started_markers": len(rows) // 2, "completed_markers": len(rows) // 2,
            "marker_pairs": len(rows) // 2, "total_markers": len(rows),
            "alternating_started_completed": True}


def validate_qualification_output(authority_sha: str) -> dict:
    if QUALIFICATION_ROOT.is_symlink() or not QUALIFICATION_ROOT.is_dir() or os.path.lexists(QUALIFICATION_PREP):
        raise RuntimeError("qualification root terminal state")
    failure = QUALIFICATION_ROOT / "failure_receipt.json"
    terminal_path = QUALIFICATION_ROOT / "terminal_receipt.json"
    if os.path.lexists(failure):
        raise RuntimeError("qualification failure receipt")
    terminal_record = regular(terminal_path)
    terminal = json.loads(terminal_path.read_text())
    if terminal.get("format") != "strict-track2-v485-v169-cache-qualification-terminal-receipt-v1" or terminal.get("passed") is not True:
        raise RuntimeError("qualification terminal semantics")
    if terminal.get("preregistration_sha256") != PREREG_SHA or terminal.get("contract_sha256") != PHASE_A_CONTRACT_SHA:
        raise RuntimeError("qualification terminal lineage")
    if terminal.get("postregistration_authority_sha256") != authority_sha or terminal.get("retry_authorized") is not False:
        raise RuntimeError("qualification authority/retry")
    for key, value in {"training_authorized": False, "cache_reuse_authorized": False, "folds": 0, "policy_updates": 0, "rl_authorized": False}.items():
        if terminal.get(key) != value:
            raise RuntimeError(f"qualification unsafe alias {key}")
    rng_proxy_record=regular(RNG_PROXY,RNG_PROXY_SHA,RNG_PROXY_BYTES)
    marker_summaries = {}
    for role_name,role in (("process_a","A"),("process_b","B")):
        receipt_path=QUALIFICATION_ROOT/role_name/"receipt.json";receipt=json.loads(receipt_path.read_text())
        count_keys=("calls","call_events_count","warning_count","raw_warning_count","rng_unchanged_count","rng_isolation_event_count","rng_isolation_delegate_started_count","rng_isolation_delegate_completed_count","rng_isolation_python_all_four_stages_equal_count","rng_isolation_numpy_all_four_stages_equal_count","rng_isolation_torch_cpu_exit_restored_count","rng_isolation_torch_cuda_exit_restored_count","durable_call_started_markers","durable_call_completed_markers")
        if receipt.get("format")!="strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1" or receipt.get("passed") is not True or receipt.get("role")!=role or any(type(receipt.get(key)) is not int or receipt.get(key)!=1000 for key in count_keys) or receipt.get("durable_call_markers_in_process_log") is not True or receipt.get("rng_proxy_source")!=rng_proxy_record or receipt.get("other_warning_count")!=0:raise RuntimeError(f"qualification {role} RNG receipt")
        marker_summaries[role] = validate_durable_process_markers(QUALIFICATION_ROOT/role_name/"process.log", role)
    report=json.loads((QUALIFICATION_ROOT/"qualification_report.json").read_text())
    summary=report.get("durable_marker_summary",{})
    if (not json_exact(report.get("per_call_rng_evidence_contract"),PER_CALL_RNG_CONTRACT)
            or summary.get("branch_count")!=2 or summary.get("calls_per_branch")!=1000
            or summary.get("total_calls")!=2000 or summary.get("total_marker_pairs")!=2000
            or summary.get("total_markers")!=4000
            or summary.get("all_pairs_alternate_started_completed") is not True
            or not json_exact(summary.get("branches"),marker_summaries)):
        raise RuntimeError("qualification exact21/durable marker summary")
    return {"terminal_receipt": terminal_record, "tree": exact_tree(QUALIFICATION_ROOT)}


def spawn_owned(command: list[str], stdout_stream, stderr_stream, state: dict) -> subprocess.Popen:
    blocked = {signal.SIGINT, signal.SIGTERM}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        process = subprocess.Popen(command, stdout=stdout_stream, stderr=stderr_stream, start_new_session=True, close_fds=True)
        if state.get("_spawn_hook") is not None:
            state["_spawn_hook"](process)
        state.update({"process": process, "pid": process.pid, "pgid": process.pid, "started": True})
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    return process


def cleanup_driver(state: dict) -> dict:
    process = state.get("process")
    pgid = state.get("pgid")
    if process is None:
        return {"started": False, "reaped": True, "group_empty": True, "returncode": None}
    if process.poll() is None:
        try: os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try: os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=TERM_GRACE_SECONDS)
    else:
        process.wait()
    try: os.killpg(pgid, 0)
    except ProcessLookupError: empty = True
    else: empty = False
    return {"started": True, "pid": process.pid, "pgid": pgid, "returncode": process.returncode,
            "reaped": process.poll() is not None, "group_empty": empty}


def close_durable(stream) -> dict:
    stream.flush(); os.fsync(stream.fileno())
    info = os.fstat(stream.fileno())
    stream.seek(0, os.SEEK_END)
    stream.close()
    return {"device": info.st_dev, "inode": info.st_ino, "closed": True}


def commit_terminal(terminal: dict, state: dict, expected_snapshot: dict, fault_hook=None) -> dict:
    if stable_snapshot(state["context"]) != expected_snapshot:
        raise RuntimeError("preterminal immutable drift")
    validate_attempt_owned(state["attempt_owned"])
    record = write_json_noreplace(ATTEMPT_ROOT / "terminal_receipt.json", terminal, state, fault_hook)
    tree = exact_tree(ATTEMPT_ROOT)
    if [row[0] for row in tree["inventory"]] != ["intent.json", "phase_a_stderr.log", "phase_a_stdout.log", "terminal_receipt.json"]:
        raise RuntimeError("terminal exact4")
    if regular(ATTEMPT_ROOT / "terminal_receipt.json") != record:
        raise RuntimeError("terminal current")
    state["committed"] = True
    return tree


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-contract", type=Path, required=True)
    parser.add_argument("--authority-contract-sha", required=True)
    parser.add_argument("--authority-contract-bytes", type=int, required=True)
    parser.add_argument("--authority-receipt", type=Path, required=True)
    parser.add_argument("--authority-receipt-sha", required=True)
    parser.add_argument("--authority-receipt-bytes", type=int, required=True)
    parser.add_argument("--launcher-source", type=Path, required=True)
    parser.add_argument("--launcher-sha", required=True)
    parser.add_argument("--launcher-bytes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--read-only-preflight", action="store_true")
    parser.add_argument("--synthetic-self-test", action="store_true")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace, hooks=None) -> int:
    hooks = hooks or {}
    context = validate_authority(args)
    if args.read_only_preflight:
        if any(os.path.lexists(p) for p in (ATTEMPT_ROOT, ATTEMPT_PREP, QUALIFICATION_ROOT, QUALIFICATION_PREP)):
            raise RuntimeError("fresh execution roots")
        if not services_healthy(health_codes()) or gpu_pids():
            raise RuntimeError("v218/GPU preflight")
        return 0
    if any(os.path.lexists(p) for p in (ATTEMPT_ROOT, ATTEMPT_PREP, QUALIFICATION_ROOT, QUALIFICATION_PREP)):
        raise RuntimeError("fresh execution roots")
    before_health = health_codes()
    if not services_healthy(before_health) or gpu_pids():
        raise RuntimeError("v218/GPU precondition")
    before = stable_snapshot(context)
    command = derive_driver_command(context)
    intent = {"format": INTENT_FORMAT, "status": "intent_committed_before_service_stop", "passed": False,
              "seed": SEED, "authority_design_contract": context["contract_record"], "authority_receipt": context["authority_record"],
              "launcher_source": context["self_record"], "driver_command": command, "driver_command_sha256": csha(command),
              "qualification_output_root": str(QUALIFICATION_ROOT), "v218_health_before": before_health,
              "authorization": AUTHORIZATION, "runtime_observation": RUNTIME_PRE, "retry_authorized": False}
    stdout_stream = stderr_stream = None
    state = {"context": context, "visible": False, "committed": False, "process": None, "started": False,
             "service_mutation_started": False, "services_stopped": False, "services_restored": False,
             "pending_signal": None, "attempt_owned": None, "_spawn_hook": hooks.get("spawn_owned_hook")}
    old_handlers = {}
    def handler(signum, _frame):
        state["pending_signal"] = signum
        raise PendingSignal(signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig); signal.signal(sig, handler)
    started_ns = time.time_ns()
    cleanup = {"started": False, "reaped": True, "group_empty": True, "returncode": None}
    stop_record = restart_record = None
    stdout_record = stderr_record = None
    driver_rc = None
    error = None
    terminal_old_mask = None
    qualification = None
    try:
        stdout_stream, stderr_stream, state["attempt_owned"] = commit_intent(intent, hooks.get("intent_fault"))
        validate_attempt_owned(state["attempt_owned"], require_initial_logs=True)
        state["service_mutation_started"] = True
        stop_record = run_service("stop")
        state["services_stopped"] = True
        stopped_health = wait_health(False)
        if not stopped_health["passed"]:
            raise RuntimeError("v218 did not stop")
        if gpu_pids():
            raise RuntimeError("GPU not empty after v218 stop")
        process = spawn_owned(command, stdout_stream, stderr_stream, state)
        if hooks.get("after_spawn"): hooks["after_spawn"](state)
        deadline = time.monotonic() + DRIVER_TIMEOUT_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            raise TimeoutError("Phase-A driver timeout")
        driver_rc = process.wait()
        cleanup = cleanup_driver(state)
        if driver_rc != 0:
            raise RuntimeError(f"Phase-A driver rc={driver_rc}")
        qualification = validate_qualification_output(context["authority_record"]["sha256"])
    except BaseException as exc:
        if terminal_old_mask is None:
            terminal_old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        error = exc
        cleanup = cleanup_driver(state)
    finally:
        if terminal_old_mask is None:
            terminal_old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        pending = signal.sigpending()
        if signal.SIGTERM in pending: state["pending_signal"] = signal.SIGTERM
        elif signal.SIGINT in pending: state["pending_signal"] = signal.SIGINT
        if stdout_stream is not None and not stdout_stream.closed:
            try:
                close_durable(stdout_stream); stdout_record = regular(ATTEMPT_ROOT / "phase_a_stdout.log")
            except BaseException as exc:
                if error is None: error = exc
        if stderr_stream is not None and not stderr_stream.closed:
            try:
                close_durable(stderr_stream); stderr_record = regular(ATTEMPT_ROOT / "phase_a_stderr.log")
            except BaseException as exc:
                if error is None: error = exc
        if state["service_mutation_started"]:
            try:
                restart_record = run_service("start")
                restored = wait_health(True)
                state["services_restored"] = restored["passed"]
                if not restored["passed"] and error is None: error = RuntimeError("v218 health restore")
            except BaseException as exc:
                if error is None: error = exc
    after_health = health_codes()
    try:
        post_gpu = gpu_pids()
    except BaseException as exc:
        if error is None: error = exc
        post_gpu = None
    success = (error is None and driver_rc == 0 and cleanup.get("reaped") is True and cleanup.get("group_empty") is True
               and state["services_restored"] and services_healthy(after_health) and post_gpu == [])
    terminal = {"format": TERMINAL_FORMAT, "status": "passed_exact_one_nested_phase_a_driver" if success else "failed_no_retry",
                "passed": success, "seed": SEED, "authority_design_contract": context["contract_record"],
                "authority_receipt": context["authority_record"], "launcher_source": context["self_record"],
                "driver_command": command, "driver_command_sha256": csha(command), "launcher_invocations": 1,
                "nested_phase_a_driver_invocations": 1 if state["started"] else 0, "direct_phase_a_driver_invocations": 0,
                "driver_returncode": driver_rc, "driver_cleanup": cleanup, "v218_stop": stop_record, "v218_restart": restart_record,
                "phase_a_stdout": stdout_record, "phase_a_stderr": stderr_record,
                "v218_health_before": before_health, "v218_health_after": after_health,
                "service_mutation_started": state["service_mutation_started"],
                "services_stopped": state["services_stopped"], "services_restored": state["services_restored"],
                "gpu_compute_pids_after": post_gpu, "qualification_output_created": QUALIFICATION_ROOT.is_dir(),
                "qualification_terminal": qualification["terminal_receipt"] if success else None,
                "qualification_tree": qualification["tree"] if success else None,
                "immutable_snapshot_before": before, "immutable_snapshot_after": stable_snapshot(context),
                "immutable_snapshots_exactly_equal": stable_snapshot(context) == before,
                "wall_nanoseconds": max(0, time.time_ns() - started_ns), "pending_signal": state["pending_signal"],
                "error_type": None if error is None else type(error).__name__, "error": None if error is None else str(error),
                "authorization_consumed": {**AUTHORIZATION, "launcher_invocations_consumed": 1,
                    "phase_a_cache_qualification_launcher_authorized": False,
                    "nested_phase_a_driver_invocations_authorized": 0,
                    "phase_a_worker_invocations_authorized": 0,
                    "runtime_delegate_calls_consumed": 2000 if success else 0,
                    "runtime_delegate_calls_authorized": 0},
                "runtime_observation": {"execution_authority_materialized": True, "phase_a_launcher_executed": True,
                    "phase_a_driver_executed": state["started"], "phase_a_worker_invocations": 2 if success else 0,
                    "rng_proxy_delegate_invocations": 2000 if success else 0, "qualification_output_created": QUALIFICATION_ROOT.is_dir(),
                    "cache_reused": False, "training_launched": False,
                    "reward_read": False, "dev_hidden_final_outcome_read": False},
                "retry_authorized": False, "cache_reuse_authorized": False, "training_authorized": False,
                "folds_authorized": 0, "policy_updates": 0, "reward_read_authorized": False,
                "dev_hidden_final_outcome_read_authorized": False}
    try:
        if ATTEMPT_ROOT.is_dir() and not state["visible"]:
            tree = commit_terminal(terminal, state, before, hooks.get("terminal_fault"))
            terminal["attempt_tree_after_terminal"] = tree  # informational local object; file stays canonical pre-field
    except BaseException as terminal_error:
        if not state["visible"]:
            raise
        if not state["committed"]:
            print(f"terminal visible but not committed: {terminal_error}", file=sys.stderr)
            return 75
    if state["committed"]:
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, terminal_old_mask)
        return 0 if success else 1
    for sig, old in old_handlers.items(): signal.signal(sig, old)
    if terminal_old_mask is not None: signal.pthread_sigmask(signal.SIG_SETMASK, terminal_old_mask)
    return 1


def synthetic_self_test() -> int:
    checks = {}
    checks["formats"] = AUTHORITY_FORMAT.endswith("-v1") and INTENT_FORMAT.endswith("-v1") and TERMINAL_FORMAT.endswith("-v1")
    checks["authorization"] = AUTHORIZATION["launcher_invocations_authorized"] == 1 and AUTHORIZATION["nested_phase_a_driver_invocations_authorized"] == 1
    checks["unsafe_false"] = all(AUTHORIZATION[key] is False for key in ("direct_phase_a_driver_authorized", "retry_authorized", "cache_reuse_authorized", "training_authorized", "reward_read_authorized", "dev_hidden_final_outcome_read_authorized"))
    checks["source_order"] = len(SOURCE_ORDER) == 27 and len(set(SOURCE_ORDER)) == 27
    checks["driver_uses_frozen_launcher"] = "--launcher-source" in derive_driver_command({"closure": {
        "fresh_phase_a_driver": {"path": "driver", "sha256": "0"*64, "logical_bytes": 1},
        "fresh_phase_a_preregistration": {"path": "prereg", "sha256": "1"*64, "logical_bytes": 1},
        "fresh_rng_proxy": {"path": "proxy", "sha256": "2"*64, "logical_bytes": 1},
        "fresh_phase_a_worker": {"path": "worker", "sha256": "b"*64, "logical_bytes": 1},
        "cache_scope_helper": {"path": "scope", "sha256": "a"*64},
        "fresh_phase_a_independent_auditor": {"path": "auditor", "sha256": "c"*64},
        "phase_a_output_materializer": {"path": "materializer", "sha256": "d"*64}, "phase_a_static_auditor": {"path": "static", "sha256": "e"*64}},
        "authority_record": {"sha256": "f"*64}})
    source = Path(__file__).read_text()
    spawn_text = source[source.index("def spawn_owned"):source.index("def cleanup_driver")]
    checks["spawn_signal_owned"] = spawn_text.index("pthread_sigmask(signal.SIG_BLOCK") < spawn_text.index("subprocess.Popen(command") < spawn_text.index("state.update") < spawn_text.index("pthread_sigmask(signal.SIG_SETMASK")
    checks["attempt_noreplace"] = "rename_dir_noreplace(ATTEMPT_PREP, ATTEMPT_ROOT)" in source
    checks["terminal_visible_committed"] = 'state["visible"] = True' in source and 'state["committed"] = True' in source
    checks["service_boundary"] = 'run_service("stop")' in source and 'run_service("start")' in source and "wait_health(True)" in source
    checks["exact4"] = '"./terminal_receipt.json"' in source
    checks["qualification_one_shot"] = "fresh execution roots" in source and "retry_authorized" in source
    checks["terminal_fault_hooks"] = all(name in source for name in ("after_terminal_link", "after_terminal_unlink", "after_terminal_dir_fsync"))
    checks["owned_prep"] = "refuse cleanup foreign member" in source
    checks["pgid_cleanup"] = "os.killpg" in source and "group_empty" in source
    checks["stdout_stderr_durable"] = "close_durable(stdout_stream)" in source and "close_durable(stderr_stream)" in source
    checks["v523_failure_ancestry"] = all(token in source for token in ("8a8fe960f79b3f026b187488babe3dc3886658520e9081a524179e728a938f5a","c67c41bdef17f340dfd3f9b6cb18a642eeadaeb0e4d97e21ab539aa2b9c90419","bda89f4244031c8b3719868a78477a5251ac864e399df0e8b47a619c8125d9f0"))
    checks["rng_proxy_binding"] = RNG_PROXY_SHA in source and WORKER_SHA in source and "rng_isolation_event_count" in source
    checks["v524_exact8"] = "materializer.validate_prereg(closure)" in source
    checks["per_call_contract_exact"] = 'cjson(contract.get("per_call_rng_evidence_contract")) != cjson(PER_CALL_RNG_CONTRACT)' in source and len(PER_CALL_RNG_CONTRACT) == 21
    checks["type_strict_authority_boundary"] = 'not json_exact(authority.get("authorization"), AUTHORIZATION)' in source and len(EXECUTION_BOUNDARY) == 8
    checks["restart_current"] = RESTART_SHA in source
    checks["interpreter_chain"] = RLPY_RESOLVED_SHA in source
    tree = ast.parse(source)
    run_service_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="run_service")
    service_popens=[node for node in ast.walk(run_service_node) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="Popen"]
    checks["service_env_unchanged"] = len(service_popens)==1 and all(keyword.arg!="env" for keyword in service_popens[0].keywords)
    checks["no_shell_driver"] = not any(
        isinstance(node, ast.Call)
        and any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in node.keywords)
        for node in ast.walk(tree)
    )
    checks["no_training"] = RUNTIME_PRE["training_launched"] is False and AUTHORIZATION["training_authorized"] is False
    checks["no_reward_hidden"] = AUTHORIZATION["reward_read_authorized"] is False and AUTHORIZATION["dev_hidden_final_outcome_read_authorized"] is False
    checks["driver_timeout"] = DRIVER_TIMEOUT_SECONDS == 9000
    checks["health_ports"] = "8005/v1/health" in source and "18084/health" in source
    checks["qualification_terminal_gate"] = 'QUALIFICATION_ROOT / "terminal_receipt.json"' in source
    checks["materializer_binding"] = AUTHORITY_MATERIALIZER_SHA in source and AUTHORITY_MATERIALIZER_BYTES == 35575
    checks["failure_repair_full_consumer"] = all(token in source for token in (
        "materializer.validate_old_state()", "v523_authority_registration_tree",
        "v523_attempt_tree", "v523_exact_source_roles_failure_root_cause"))
    checks["authority_schema_counts"] = len(CONTRACT_TOP_KEYS) == 35 and len(CHECK_KEYS) == 33 and len(SOURCE_ORDER) == 27
    checks["terminal_signal_mask"] = "terminal_old_mask = signal.pthread_sigmask" in source
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    print(json.dumps({"passed": True, "check_count": len(checks), "checks_sha256": csha(checks)}, sort_keys=True))
    return 0


def main() -> int:
    if "--synthetic-self-test" in sys.argv:
        return synthetic_self_test()
    return execute(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
