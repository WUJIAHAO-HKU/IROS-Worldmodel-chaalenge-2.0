#!/usr/bin/env python3
"""Exact-once Stage-A transport for the v540 public-S1 zero-update poststage-owned-prep repair authority."""
from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS = ROOT / "pipeline/scripts"
RLPY = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
SELF_PATH = SCRIPTS / "invoke_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_materializer_once.py"
SCRIPT_PATH = Path("/root/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_materialize_once.sh")
DEPLOYMENT_RECORD = SCRIPTS / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_materialization_transport_deployment_record.json"
CONTRACT = SCRIPTS / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_contract.json"
CONTRACT_SHA = "f565e6c49a4d30fb9ae329f4e259010c85359fde6fd8479f3743bdabdc45ef22"
CONTRACT_BYTES = 154933
MATERIALIZER = SCRIPTS / "materialize_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority.py"
MATERIALIZER_SHA = "6e2b61045a499a1a76b65d57780b915604101ebe52a373e801023943d3baf1b3"
MATERIALIZER_BYTES = 30925
EVALUATOR = SCRIPTS / "evaluate_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py"
EVALUATOR_SHA = "d1c2a65f087cca7175d9638219e6174bb91d5afefbbd71f7290b72387f3cd911"
EVALUATOR_BYTES = 67828
AUDITOR = SCRIPTS / "audit_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py"
AUDITOR_SHA = "e5ce8c289f108eb0aaf593a21afb2164fc2838ef915e4a49e909a85078227e46"
AUDITOR_BYTES = 5528
LAUNCHER = SCRIPTS / "launch_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py"
LAUNCHER_SHA = "9f6d272350907eb072dfb20da536128cf85efed5e9b50325e1a9c273952d3afb"
LAUNCHER_BYTES = 4480
PREREGISTRATION = SCRIPTS / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_preregistration.json"
PREREGISTRATION_SHA = "370f6991ad0f85e966467a594efc4cbb7378cf6a82f5982bdd7ff077cfbf8b6c"
PREREGISTRATION_BYTES = 81186
MANIFEST = SCRIPTS / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_manifest.json"
MANIFEST_SHA = "df9196600ef9cb728e9300dabacf9fccf10f2a55501c86f9076eda5473b40d29"
MANIFEST_BYTES = 83407
FORENSIC = SCRIPTS / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_absence_failure_forensic.json"
FORENSIC_SHA = "ce412975394a8107853945c486e6a859984fd550ba9277f00ae013fbf0d22b43"
FORENSIC_BYTES = 7541
OLD_HELPER = SCRIPTS / "invoke_v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_materializer_once.py"
OLD_HELPER_SHA = "60ce97d73a15841e98ec31554eb383be9be3379411b995ab0cc7075b3b4b0d88"
OLD_HELPER_BYTES = 81868
OLD_SCRIPT = Path("/root/v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_materialize_once.sh")
OLD_SCRIPT_SHA = "3f8c74ffec2756fcbe0d9412a8d1031693e157a670d4b2b03e7106b98b28da2f"
OLD_SCRIPT_BYTES = 4217
OLD_RECORD = SCRIPTS / "v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_materialization_transport_deployment_record.json"
OLD_RECORD_SHA = "ae893fde2b0c3cfebe32e089361568e6f28cab6c022d172c2487743d3123711b"
OLD_RECORD_BYTES = 968
AUTHORITY_ROOT = J / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_seed1671_20260828"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".authority-prep")
EVIDENCE_ROOT = J / "v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_materialization_evidence_seed1671_20260828"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
EXACT5 = ["argv.json", "intent.json", "materializer_stderr.log", "materializer_stdout.log", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])
AT_FDCWD = -100
RENAME_NOREPLACE = 1
FROZEN_FINAL8_SHA256 = (CONTRACT_SHA, MATERIALIZER_SHA, PREREGISTRATION_SHA, MANIFEST_SHA,
                       EVALUATOR_SHA, AUDITOR_SHA, LAUNCHER_SHA, FORENSIC_SHA)
EXPECTED_CURRENT_ABSENCE_KEYS = ["attempt_prep", "attempt_root", "authority_prep", "output_prep", "output_root"]
EXPECTED_CURRENT_ABSENCE_PATHS = {
    "attempt_prep": "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_gate_attempt_seed1671_20260828.attempt-prep",
    "attempt_root": "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_gate_attempt_seed1671_20260828",
    "authority_prep": "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_seed1671_20260828.authority-prep",
    "output_prep": "/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828.output-prep",
    "output_root": "/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828",
}
EXPECTED_CURRENT_ABSENCE_MAP_SHA256 = "98f9acd9e7d448d264a04b2077c85fe905995d5c85fb8fcbc75331ac3f936add"


def deployment_plan() -> dict:
    final8 = [
        {"role": "authority_design_contract", "path": str(CONTRACT), "sha256": CONTRACT_SHA, "logical_bytes": CONTRACT_BYTES},
        {"role": "authority_materializer", "path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
        {"role": "public_s1_zero_update_preregistration", "path": str(PREREGISTRATION), "sha256": PREREGISTRATION_SHA, "logical_bytes": PREREGISTRATION_BYTES},
        {"role": "public_s1_zero_update_manifest", "path": str(MANIFEST), "sha256": MANIFEST_SHA, "logical_bytes": MANIFEST_BYTES},
        {"role": "public_s1_evaluator", "path": str(EVALUATOR), "sha256": EVALUATOR_SHA, "logical_bytes": EVALUATOR_BYTES},
        {"role": "public_s1_independent_auditor", "path": str(AUDITOR), "sha256": AUDITOR_SHA, "logical_bytes": AUDITOR_BYTES},
        {"role": "public_s1_launcher", "path": str(LAUNCHER), "sha256": LAUNCHER_SHA, "logical_bytes": LAUNCHER_BYTES},
        {"role": "v539_poststage_owned_output_prep_absence_failure_forensic", "path": str(FORENSIC), "sha256": FORENSIC_SHA, "logical_bytes": FORENSIC_BYTES},
    ]
    return {
        "fresh_final8_noreplace": final8,
        "fresh_final8_volatile_sources": [{**row, "source_path": "/dev/shm/" + Path(row["path"]).name} for row in final8],
        "fresh_noncyclic_transport_triple_external_noreplace_required": True,
        "fresh_transport_invocations_before_normal": 0,
        "canonical_readback_exact_required": True,
        "atomic_source_publication_contract": {
            "target_prestate_absent_required": True, "temporary_member_open_flags": ["O_RDWR", "O_CREAT", "O_EXCL", "O_NOFOLLOW"],
            "held_descriptor_through_link_and_owned_unlink": True, "write_all_required": True, "file_fsync_required": True,
            "pread_exact_sha_bytes_and_eof_required": True, "fstat_lstat_dev_inode_identity_required": True,
            "hardlink_noreplace_then_owned_temp_unlink": True, "parent_directory_fsync_required": True,
            "canonical_exact_readback_required": True, "foreign_replacement_preserved_and_rejected": True,
            "manifest_large_file_bytes": MANIFEST_BYTES,
        },
    }


class ControlledSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"controlled signal {signum}")
        self.signum = signum


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def csha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: tuple[str, int] | None = None) -> dict:
    if path != path.resolve() or path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(f"regular: {path}")
    row = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if expected is not None and (row["sha256"], row["logical_bytes"]) != expected:
        raise RuntimeError(f"record mismatch: {path}")
    return row


def read_json_exact(path: Path, digest: str, logical_bytes: int) -> dict:
    regular(path, (digest, logical_bytes))
    return json.loads(path.read_text(encoding="utf-8"))


def validate_failure_forensic() -> dict:
    value = read_json_exact(FORENSIC, FORENSIC_SHA, FORENSIC_BYTES)
    expected_top = ["authority_state", "classification", "current_readonly_state", "failure",
                    "failure_partition", "format", "historical_transport", "model_rng_cleanup",
                    "root_cause", "source_review_explanation", "status", "successor_boundary"]
    boundary = value.get("successor_boundary", {})
    partition = value.get("failure_partition", {})
    root_cause = value.get("root_cause", {})
    authority = value.get("authority_state", {})
    cleanup = value.get("model_rng_cleanup", {})
    if (list(value) != expected_top
            or value.get("format") != "strict-track2-v540-v539-public-s1-zero-update-poststage-owned-output-prep-absence-failure-forensic-v1"
            or value.get("status") != "failed_no_retry_owned_output_prep_phase_misclassified_as_required_absence"
            or boundary.get("fresh_authority_required") is not True
            or boundary.get("old_transport_retry_authorized") is not False
            or boundary.get("old_authority_consumption_mutation_authorized") is not False
            or type(partition.get("public_s1_zero_update_boundary_invocations_non_durable")) is not int
            or partition.get("public_s1_zero_update_boundary_invocations_non_durable") != 1
            or partition.get("public_s1_evaluator_invocations_non_durable") != 1
            or partition.get("runtime_package_load_invocations_non_durable") != 1
            or partition.get("public_rows_computed_non_durable") != 44
            or partition.get("zero_update_gate_invocations_non_durable") != 1
            or partition.get("auditor_import_invocations_non_durable") != 1
            or partition.get("training_invocations") != 0
            or root_cause.get("failure_phase") != "after_runtime_44_public_rows_zero_update_auditor_and_exact8_staging_before_noreplace_output_commit"
            or root_cause.get("poststage_validate_documents_require_fresh_false_still_applied_unconditional_current_absence_gate") is not True
            or authority.get("sha256") != "0b43b24c004c6a00975d3b14ece7167b312b9735185b1241454e39f856894be4"
            or type(authority.get("authorization_public_s1_zero_update_boundary_invocations_consumed")) is not int
            or authority.get("authorization_public_s1_zero_update_boundary_invocations_consumed") != 0
            or cleanup.get("owned_staged_exact8_removed") is not True
            or cleanup.get("owned_staging_directory_removed") is not True):
        raise RuntimeError("v540 failure forensic exact")
    return regular(FORENSIC, (FORENSIC_SHA, FORENSIC_BYTES))


def expected_transport_deployment(helper_record: dict, script_record: dict) -> dict:
    return {
        "format": "strict-track2-v540-stage-a-transport-pair-deployment-record-v1",
        "status": "preregistered_atomic_noreplace_transport_pair_pending_external_deployment",
        "classification": "design_only_non_authority_non_execution_transport_pair_manifest",
        "deployment_authority": "external_atomic_noreplace_deployer_required",
        "deployment_executed": False,
        "pair_exact_count": 2,
        "targets": [
            {"role": "transport_helper", **helper_record},
            {"role": "transport_script", **script_record},
        ],
        "canonical_readback_required": True,
        "overwrite_authorized": False,
    }


def validate_transport_deployment_record(helper_record: dict, script_record: dict) -> dict:
    record = regular(DEPLOYMENT_RECORD)
    raw = DEPLOYMENT_RECORD.read_bytes()
    value = json.loads(raw)
    expected = expected_transport_deployment(helper_record, script_record)
    if raw != cbytes(value) or cbytes(value) != cbytes(expected):
        raise RuntimeError("transport deployment record")
    return record


def exact_tree(root: Path) -> dict:
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root: {root}")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"tree member: {path}")
        if path.is_file():
            if not stat.S_ISREG(os.lstat(path).st_mode):
                raise RuntimeError(f"tree member: {path}")
            inventory.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree member: {path}")
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in inventory).encode()
    return {
        "root": str(root), "inventory": inventory, "file_count": len(inventory),
        "logical_file_bytes": sum(row[2] for row in inventory),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": csha(inventory),
    }


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise RuntimeError("short write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def terminal_write(descriptor: int, payload) -> int:
    return os.write(descriptor, payload)


def terminal_pread(descriptor: int, size: int, offset: int) -> bytes:
    return os.pread(descriptor, size, offset)


def terminal_link(source: Path, target: Path) -> None:
    os.link(source, target, follow_symlinks=False)


def unlink_terminal_temp(path: Path) -> None:
    path.unlink()


def fsync_terminal_parent(path: Path) -> None:
    fsync_dir(path)


def held_terminal_record(descriptor: int, path: Path, identity: tuple[int, int]) -> dict:
    descriptor_metadata = os.fstat(descriptor)
    path_metadata = os.lstat(path)
    if (not stat.S_ISREG(descriptor_metadata.st_mode) or path.is_symlink()
            or not stat.S_ISREG(path_metadata.st_mode)
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != identity
            or (path_metadata.st_dev, path_metadata.st_ino) != identity):
        raise RuntimeError("terminal held member identity")
    size = descriptor_metadata.st_size
    chunks = []
    offset = 0
    while offset < size:
        block = terminal_pread(descriptor, min(1 << 20, size - offset), offset)
        if not block:
            raise RuntimeError("terminal held member short pread")
        chunks.append(block)
        offset += len(block)
    if terminal_pread(descriptor, 1, size) != b"":
        raise RuntimeError("terminal held member EOF")
    payload = b"".join(chunks)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "logical_bytes": len(payload)}


def owned_terminal_cleanup(descriptor: int, path: Path, identity: tuple[int, int]) -> None:
    first = held_terminal_record(descriptor, path, identity)
    second = held_terminal_record(descriptor, path, identity)
    if first != second:
        raise RuntimeError("terminal cleanup record drift")
    unlink_terminal_temp(path)
    fsync_dir(path.parent)


def atomic_json_noreplace(path: Path, value: object, visible_hook, committed_hook, fixture_hook=None) -> dict | None:
    temporary = path.with_name("." + path.name + ".noreplace-tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise FileExistsError(path)
    payload = cbytes(value)
    descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    descriptor_metadata = os.fstat(descriptor)
    temporary_identity = (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    if not stat.S_ISREG(descriptor_metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError("terminal temp not regular")
    linked = False
    try:
        view = memoryview(payload)
        while view:
            count = terminal_write(descriptor, view)
            if count <= 0:
                raise RuntimeError("terminal short write")
            view = view[count:]
        os.fsync(descriptor)
        if fixture_hook is not None:
            fixture_hook("before_first_identity", temporary, descriptor)
        temporary_record = held_terminal_record(descriptor, temporary, temporary_identity)
        expected_record = {"path": str(temporary), "sha256": hashlib.sha256(payload).hexdigest(),
                           "logical_bytes": len(payload)}
        if temporary_record != expected_record:
            raise RuntimeError("terminal temp content")
        if fixture_hook is not None:
            fixture_hook("after_record", temporary, descriptor)
        if held_terminal_record(descriptor, temporary, temporary_identity) != temporary_record:
            raise RuntimeError("terminal temp ownership drift")
        terminal_link(temporary, path)
        linked = True
        target_record = held_terminal_record(descriptor, path, temporary_identity)
        target_record["path"] = str(temporary)
        if target_record != temporary_record:
            raise RuntimeError("terminal target identity")
        visible_hook()
        if fixture_hook is not None:
            fixture_hook("after_link", temporary, descriptor)
        if held_terminal_record(descriptor, temporary, temporary_identity) != temporary_record:
            raise RuntimeError("terminal pre-unlink ownership drift")
        unlink_terminal_temp(temporary)
        committed_hook()
        try:
            if fixture_hook is not None:
                fixture_hook("postcommit", temporary, descriptor)
            fsync_terminal_parent(path.parent)
            current_record = held_terminal_record(descriptor, path, temporary_identity)
            if current_record != {"path": str(path), "sha256": temporary_record["sha256"],
                                  "logical_bytes": temporary_record["logical_bytes"]}:
                raise RuntimeError("terminal postcommit current")
        except BaseException as error:
            # Choice-A visibility commit: after the owned temp unlink syscall
            # succeeds, parent fsync is best-effort and cannot revoke success.
            return {"stage": "atomic_terminal_postcommit", "error_type": type(error).__name__, "error": str(error)}
        return None
    except BaseException:
        if not linked and os.path.lexists(temporary):
            owned_terminal_cleanup(descriptor, temporary, temporary_identity)
        raise
    finally:
        os.close(descriptor)


def close_fsync(stream) -> None:
    if stream is None or stream.closed:
        return
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()


def fsync_stream(stream) -> None:
    if stream is None or stream.closed:
        raise RuntimeError("owned stream closed")
    stream.flush()
    os.fsync(stream.fileno())


def assert_owned_fd(stream, path: Path, identity: tuple[int, int]) -> None:
    descriptor_metadata = os.fstat(stream.fileno())
    path_metadata = os.lstat(path)
    if (not stat.S_ISREG(descriptor_metadata.st_mode) or path.is_symlink()
            or not stat.S_ISREG(path_metadata.st_mode)
            or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != identity
            or (path_metadata.st_dev, path_metadata.st_ino) != identity):
        raise RuntimeError("owned log identity drift")


def owned_member(path: Path) -> dict:
    metadata = os.lstat(path)
    return {"identity": (metadata.st_dev, metadata.st_ino), "record": regular(path)}


def cleanup_owned_prep(path: Path, identity: tuple[int, int] | None,
                       owned_members: dict[str, dict]) -> None:
    if identity is None or not path.is_dir() or path.is_symlink():
        return
    observed = os.lstat(path)
    if (observed.st_dev, observed.st_ino) != identity:
        return
    children = {child.name: child for child in path.iterdir()}
    if set(children) != set(owned_members):
        raise RuntimeError("foreign prep member set")
    for name, child in children.items():
        metadata = os.lstat(child)
        expected = owned_members[name]
        if (child.is_symlink() or not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != expected["identity"]
                or regular(child) != expected["record"]):
            raise RuntimeError("foreign prep member identity/record")
    for child in children.values():
        child.unlink()
    path.rmdir()
    fsync_dir(path.parent)


def group_empty(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def terminate(process: subprocess.Popen | None) -> dict:
    result = {"term_sent": False, "kill_sent": False, "reaped": process is None, "group_empty": process is None}
    if process is None:
        return result
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            result["term_sent"] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                result["kill_sent"] = True
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=3)
        result["reaped"] = True
    except subprocess.TimeoutExpired:
        result["reaped"] = False
    result["group_empty"] = group_empty(process.pid)
    return result


def records_current(value: object) -> bool:
    found = []
    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"path", "sha256", "logical_bytes"} and isinstance(item["path"], str):
                found.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return bool(found) and all(regular(Path(row["path"])) == row for row in found)


def service_health() -> dict:
    result = {}
    for name, url in (("8005_v1_health", "http://127.0.0.1:8005/v1/health"),
                      ("18084_health", "http://127.0.0.1:18084/health")):
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            try:
                model = json.loads(body)
            except Exception:
                model = None
            result[name] = {"http_code": response.status, "body_sha256": hashlib.sha256(body).hexdigest(),
                            "body_bytes": len(body), "json_model": model}
    if not all(row["http_code"] == 200 and row["body_bytes"] > 0 for row in result.values()):
        raise RuntimeError("service health")
    return result


def gpu_compute_pids() -> list[int]:
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False, timeout=20)
    if process.returncode != 0:
        raise RuntimeError("nvidia-smi")
    return sorted({int(line.strip()) for line in process.stdout.splitlines() if line.strip()})


def forbidden_entrypoints() -> set[str]:
    return {
        str(LAUNCHER), str(EVALUATOR), str(AUDITOR),
        str(SCRIPTS / "launch_v524_v523_phase_a_cache_qualification.py"),
        str(SCRIPTS / "generate_v524_v523_phase_a_cache_qualification.py"),
        str(SCRIPTS / "generate_v524_v523_phase_a_cache_qualification_worker.py"),
        str(SCRIPTS / "v520_v519_rng_isolated_v169_runtime_proxy.py"),
        str(SCRIPTS / "reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py"),
        str(SCRIPTS / "train_v482_temporal8_residual_5fold.py"),
    }


def classify_forbidden_argv(argv: list[str]) -> list[str]:
    return sorted(set(argv) & forbidden_entrypoints())


def phase_a_or_reconciler_pids() -> list[dict]:
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [part.decode(errors="replace") for part in
                    (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        matches = classify_forbidden_argv(argv)
        if matches:
            rows.append({"pid": int(entry.name), "argv": argv, "exact_entrypoint_matches": matches})
    return sorted(rows, key=lambda row: row["pid"])


def stable_absences(contract: dict) -> dict:
    expected = {key: {"absent": True, "path": EXPECTED_CURRENT_ABSENCE_PATHS[key]}
                for key in EXPECTED_CURRENT_ABSENCE_KEYS}
    rows = contract.get("current_absences_after_authority")
    if (not isinstance(rows, dict) or list(rows) != EXPECTED_CURRENT_ABSENCE_KEYS
            or set(rows) != set(EXPECTED_CURRENT_ABSENCE_KEYS)
            or csha(rows) != EXPECTED_CURRENT_ABSENCE_MAP_SHA256
            or cbytes(rows) != cbytes(expected)):
        raise RuntimeError("contract absence map order/bijection/hash")
    absences = {}
    seen_paths = set()
    for key, row in rows.items():
        if (not isinstance(row, dict) or list(row) != ["absent", "path"] or set(row) != {"absent", "path"}
                or type(row.get("absent")) is not bool or row["absent"] is not True
                or type(row.get("path")) is not str or row["path"] != EXPECTED_CURRENT_ABSENCE_PATHS[key]
                or row["path"] in seen_paths):
            raise RuntimeError("contract absence schema")
        seen_paths.add(row["path"])
        path = Path(row["path"])
        absences[key] = {"path": str(path), "absent": not os.path.lexists(path)}
    if not all(row["absent"] for row in absences.values()):
        raise RuntimeError("contract current absence")
    if list(absences) != EXPECTED_CURRENT_ABSENCE_KEYS or set(seen_paths) != set(EXPECTED_CURRENT_ABSENCE_PATHS.values()):
        raise RuntimeError("contract absence observed bijection")
    return absences


def immutable_snapshot(contract: dict, helper_record: dict, script_record: dict) -> dict:
    sources = {role: regular(Path(row["path"]), (row["sha256"], row["logical_bytes"]))
               for role, row in contract["source_closure"].items()}
    trees = {}
    for key, row in contract.items():
        if isinstance(row, dict) and set(row) == {"root", "inventory", "file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"}:
            observed = exact_tree(Path(row["root"]))
            if observed != row:
                raise RuntimeError(f"contract tree drift: {key}")
            trees[key] = observed
    absences = stable_absences(contract)
    deployment_record = validate_transport_deployment_record(helper_record, script_record)
    return {
        "contract": regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES)),
        "materializer": regular(MATERIALIZER, (MATERIALIZER_SHA, MATERIALIZER_BYTES)),
        "transport_helper": regular(SELF_PATH, (helper_record["sha256"], helper_record["logical_bytes"])),
        "transport_script": regular(SCRIPT_PATH, (script_record["sha256"], script_record["logical_bytes"])),
        "transport_deployment_record": deployment_record,
        "v539_poststage_owned_output_prep_absence_failure_forensic": validate_failure_forensic(),
        "sources": sources, "trees": trees, "absences": absences,
        "service_health": service_health(), "gpu_compute_pids": gpu_compute_pids(),
        "phase_a_or_reconciler_pids": phase_a_or_reconciler_pids(), "deployment_plan": deployment_plan(),
    }


def validate_materializer_contract(module, contract: dict, materializer_record: dict) -> None:
    seed = contract.get("seed")
    active = {
        "authority_materializer": {"path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
        "public_s1_zero_update_preregistration": {"path": str(PREREGISTRATION), "sha256": PREREGISTRATION_SHA, "logical_bytes": PREREGISTRATION_BYTES},
        "public_s1_zero_update_manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA, "logical_bytes": MANIFEST_BYTES},
        "public_s1_evaluator": {"path": str(EVALUATOR), "sha256": EVALUATOR_SHA, "logical_bytes": EVALUATOR_BYTES},
        "public_s1_independent_auditor": {"path": str(AUDITOR), "sha256": AUDITOR_SHA, "logical_bytes": AUDITOR_BYTES},
        "public_s1_launcher": {"path": str(LAUNCHER), "sha256": LAUNCHER_SHA, "logical_bytes": LAUNCHER_BYTES},
    }
    schema_checks = {
        "contract_path": module.CONTRACT == CONTRACT,
        "authority_root": module.AUTH_ROOT == AUTHORITY_ROOT,
        "contract_top": set(contract) == module.CONTRACT_TOP_KEYS,
        "format": contract.get("format") == module.CONTRACT_FORMAT,
        "status": contract.get("status") == module.CONTRACT_STATUS,
        "seed": type(seed) is int and seed == 1671,
        "materializer_record": contract.get("authority_materializer_source") == materializer_record,
        "active_records": not any(contract.get("source_closure", {}).get(role) != row for role, row in active.items()),
        "active_role_order": contract.get("active_source_role_order") == module.ACTIVE_SOURCE_ROLE_ORDER,
        "active_paths": contract.get("active_source_paths") == module.ACTIVE_SOURCE_PATHS,
        "contract_active_records_exact6": contract.get("active_source_records") == active,
        "authority_top_count": len(contract.get("authority_receipt_contract", {}).get("top_keys", [])) == 69,
        "check_count": len(module.CHECK_KEYS) == 31,
        "source_counts": len(module.SOURCE_ROLES) == 34 and len(module.AUTHORITY_SOURCE_ROLES) == 35,
        "source_order": contract.get("source_role_order") == module.SOURCE_ROLES,
        "source_aliases": contract.get("source_aliases") == module.SOURCE_ALIASES,
        "v534_durable_transition_current": (
            contract.get("v534_actual_oof_durable_transition", {}).get("exact_count") == 11
            and type(contract.get("v534_actual_oof_durable_transition", {}).get("durable_partition", {}).get("actual_oof_execution_boundary_invocations")) is int
            and contract.get("v534_actual_oof_durable_transition", {}).get("durable_partition", {}).get("actual_oof_execution_boundary_invocations") == 1
            and type(contract.get("v534_actual_oof_durable_transition", {}).get("old_authority_receipt", {}).get("consumed")) is int
            and contract.get("v534_actual_oof_durable_transition", {}).get("old_authority_receipt", {}).get("consumed") == 0
            and contract.get("v534_actual_oof_durable_transition", {}).get("durable_partition", {}).get("retry_authorized") is False),
        "v539_failed_transition_no_retry": (
            contract.get("v539_poststage_owned_output_prep_failure_transition", {}).get("failure_forensic")
                == {"path": str(FORENSIC), "sha256": FORENSIC_SHA, "logical_bytes": FORENSIC_BYTES}
            and type(contract.get("v539_poststage_owned_output_prep_failure_transition", {}).get("old_authority_receipt", {}).get("consumed")) is int
            and contract.get("v539_poststage_owned_output_prep_failure_transition", {}).get("old_authority_receipt", {}).get("consumed") == 0
            and contract.get("v539_poststage_owned_output_prep_failure_transition", {}).get("retry_authorized") is False
            and contract.get("v539_poststage_owned_output_prep_failure_transition", {}).get("staged_exact8_before_failure") is True
            and contract.get("public_s1_output_contract", {}).get("phase_validation", {}).get("generic_poststage_recheck_of_output_prep_absence_authorized") is False),
        "lineage_exact": cbytes(contract.get("lineage")) == cbytes(module.LINEAGE),
        "public_s1_zero_update_contracts": (
            cbytes(contract.get("public_s1_input_contract")) == cbytes(read_json_exact(PREREGISTRATION, PREREGISTRATION_SHA, PREREGISTRATION_BYTES).get("public_s1_input_contract"))
            and cbytes(contract.get("zero_update_contract")) == cbytes(read_json_exact(MANIFEST, MANIFEST_SHA, MANIFEST_BYTES).get("zero_update_contract"))
            and contract.get("hidden_input_denial_contract", {}).get("hidden_private_final_reward_inputs_authorized") is False),
    }
    if not all(schema_checks.values()):
        raise RuntimeError("materializer/contract schema " + json.dumps(schema_checks, sort_keys=True))
    if (cbytes(contract.get("authorization")) != cbytes(module.AUTHORIZATION)
            or csha(contract.get("runtime_observation")) != csha(module.RUNTIME)
            or csha(contract.get("execution_boundary")) != csha(module.EXECUTION_BOUNDARY)):
        raise RuntimeError("v535 public-S1 zero-update authority boundary")
    for role, row in contract["source_closure"].items():
        if regular(Path(row["path"]), (row["sha256"], row["logical_bytes"])) != row:
            raise RuntimeError(f"source closure current: {role}")
    context_contract = regular(Path(module.CONTRACT))
    context_materializer = regular(Path(module.SELF))
    module.validate_context(argparse.Namespace(
        contract=Path(module.CONTRACT),
        contract_sha=context_contract["sha256"], contract_bytes=context_contract["logical_bytes"],
        materializer_source=Path(module.SELF),
        materializer_sha=context_materializer["sha256"], materializer_bytes=context_materializer["logical_bytes"],
        authority_root=Path(module.AUTH_ROOT), read_only_preflight=True))


def load_materializer(contract_source: Path = CONTRACT, materializer_source: Path = MATERIALIZER,
                      module_overrides: dict | None = None):
    contract_record = regular(contract_source, (CONTRACT_SHA, CONTRACT_BYTES))
    materializer_record = regular(materializer_source, (MATERIALIZER_SHA, MATERIALIZER_BYTES))
    if contract_source != CONTRACT:
        contract_record = {**contract_record, "path": str(CONTRACT)}
    if materializer_source != MATERIALIZER:
        materializer_record = {**materializer_record, "path": str(MATERIALIZER)}
    contract = json.loads(contract_source.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("v540_materializer", materializer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in (module_overrides or {}).items():
        setattr(module, name, value)
    validate_materializer_contract(module, contract, materializer_record)
    return module, contract, contract_record, materializer_record


def command() -> list[str]:
    return [str(RLPY), str(MATERIALIZER), "--contract", str(CONTRACT),
            "--contract-sha", CONTRACT_SHA, "--contract-bytes", str(CONTRACT_BYTES),
            "--materializer-source", str(MATERIALIZER), "--materializer-sha", MATERIALIZER_SHA,
            "--materializer-bytes", str(MATERIALIZER_BYTES), "--authority-root", str(AUTHORITY_ROOT)]


def read_fd_all_exact(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    payload = os.pread(descriptor, size + 1, 0)
    if len(payload) != size or os.pread(descriptor, 1, size) != b"":
        raise RuntimeError("stdout EOF")
    return payload


def write_all(descriptor: int, payload: bytes, writer=os.write) -> None:
    offset = 0
    while offset < len(payload):
        count = writer(descriptor, payload[offset:])
        if count is None or isinstance(count, bool) or count <= 0 or count > len(payload) - offset:
            raise RuntimeError("stdout short/zero/invalid write")
        offset += count


def transport_stdout_exact2(stream, path: Path, identity: tuple[int, int],
                            authority_record: dict, authority_tree: dict,
                            writer=os.write) -> tuple[list[dict], dict]:
    assert_owned_fd(stream, path, identity)
    descriptor = stream.fileno()
    try:
        native = read_fd_all_exact(descriptor)
        native_lines = native.splitlines(keepends=True)
        if len(native_lines) != 1 or not native.endswith(b"\n") or native.endswith(b"\n\n"):
            raise RuntimeError("materializer native stdout exact1")
        try:
            native_json = json.loads(native.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("materializer native stdout JSON") from error
        if (set(native_json)!={"authority_root","passed","publication","receipt_bytes","receipt_sha256"}
                or native_json.get("passed") is not True
                or native_json.get("authority_root")!=str(AUTHORITY_ROOT)
                or native_json.get("receipt_sha256")!=authority_record["sha256"]
                or type(native_json.get("receipt_bytes")) is not int
                or native_json.get("receipt_bytes")!=authority_record["logical_bytes"]
                or native_json.get("publication",{}).get("visibility_committed") is not True):
            raise RuntimeError("materializer native stdout protocol")
        native_record = {
            "line_origin": "materializer_native_exact1", "logical_bytes": len(native),
            "sha256": hashlib.sha256(native).hexdigest(), "eof_exact": True,
            "parsed": native_json,
        }
        helper_line = {"line_origin": "transport_helper", "passed": True,
              "transport_helper_invocations": 1, "authority_materializer_invocations": 1,
              "public_s1_zero_update_boundary_invocations": 0,
              "public_s1_evaluator_invocations": 0, "zero_update_gate_invocations": 0,
              "public_s1_independent_auditor_invocations": 0,
              "v534_actual_oof_durable_boundary_invocations": 1,
              "actual_oof_execution_invocations": 0,
              "failed_v539_transport_helper_normal_entries_non_durable": 1,
              "failed_v539_transport_helper_popen_invocations": 0,
             "phase_a_launcher_invocations": 0,
             "phase_a_driver_invocations": 0, "phase_a_worker_invocations": 0,
             "rng_proxy_delegate_invocations": 0, "oof_model_execution_invocations": 0,
             "phase_a_replay_invocations": 0, "training_invocations": 0,
             "cache_reuse_invocations": 0, "reward_read_invocations": 0,
             "dev_hidden_final_outcome_read_invocations": 0,
             "submission_invocations": 0, "committed_success": True,
             "authority_receipt": authority_record, "authority_registration_tree": authority_tree}
        lines = [native_json, helper_line]
        helper_payload = cbytes(helper_line)
        os.lseek(descriptor, 0, os.SEEK_END)
        write_all(descriptor, helper_payload, writer)
        os.fsync(descriptor)
        observed = read_fd_all_exact(descriptor)
        if observed != native + helper_payload or observed.splitlines(keepends=True) != [native, helper_payload]:
            raise RuntimeError("transport stdout exact2 readback")
        assert_owned_fd(stream, path, identity)
        return lines, native_record
    finally:
        pass


def rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 unavailable")
    result = renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(target), RENAME_NOREPLACE)
    if result != 0:
        error = ctypes.get_errno()
        if error == 17:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), str(target))


def validate_authority_aliases(receipt: dict, module) -> None:
    if set(receipt.get("source_aliases", {})) != set(module.AUTHORITY_SOURCE_ROLES):
        raise RuntimeError("authority source alias keyset")
    for role in module.AUTHORITY_SOURCE_ROLES:
        alias = module.SOURCE_ALIASES[role]
        if receipt.get(alias) != receipt.get("source_closure", {}).get(role):
            raise RuntimeError(f"authority top-level source alias:{role}")


def validate_success(module, contract: dict, returncode: int, cleanup: dict,
                     stdout_stream, stdout_identity: tuple[int, int], writer=os.write) -> dict:
    if returncode != 0 or not cleanup["reaped"] or not cleanup["group_empty"]:
        raise RuntimeError(f"materializer child rc {returncode}")
    if regular(EVIDENCE_ROOT / "materializer_stderr.log")["logical_bytes"] != 0:
        raise RuntimeError("materializer native stderr must be empty")
    stdout_path = EVIDENCE_ROOT / "materializer_stdout.log"
    receipt_record = regular(AUTHORITY_ROOT / "authority_receipt.json")
    receipt = json.loads(Path(receipt_record["path"]).read_text(encoding="utf-8"))
    tree = exact_tree(AUTHORITY_ROOT)
    receipt_schema = contract["authority_receipt_contract"]
    authority_active_records = receipt.get("active_source_records")
    if (os.path.lexists(AUTHORITY_PREP)
            or sorted(receipt) != receipt_schema["top_keys"] or len(receipt) != 69
            or receipt.get("format") != module.OUTPUT_FORMAT
            or receipt.get("status") != module.OUTPUT_STATUS or receipt.get("passed") is not True
            or len(receipt.get("checks", {})) != 31
            or set(receipt.get("checks", {})) != set(module.CHECK_KEYS)
            or not all(type(value) is bool and value is True for value in receipt["checks"].values())
            or receipt.get("check_key_set_sha256") != csha(sorted(module.CHECK_KEYS))
            or receipt.get("checks_sha256") != csha(receipt["checks"])
            or len(receipt.get("source_closure", {})) != 35
            or receipt.get("source_role_order") != module.AUTHORITY_SOURCE_ROLES
            or receipt.get("source_aliases") != module.SOURCE_ALIASES
            or receipt.get("active_source_role_order") != module.ACTIVE_SOURCE_ROLE_ORDER
            or receipt.get("active_source_paths") != module.ACTIVE_SOURCE_PATHS
            or not isinstance(authority_active_records, dict)
            or set(authority_active_records) != set(module.ACTIVE_SOURCE_ROLE_ORDER)
            or any(authority_active_records.get(role) != receipt.get("source_closure", {}).get(role)
                   for role in module.ACTIVE_SOURCE_ROLE_ORDER)
            or receipt_schema.get("active_source_role_order_exact") != module.ACTIVE_SOURCE_ROLE_ORDER
            or receipt_schema.get("active_source_paths_exact") != module.ACTIVE_SOURCE_PATHS
            or receipt_schema.get("contract_active_source_records_exact") != contract.get("active_source_records")
            or type(receipt_schema.get("contract_active_source_records_count")) is not int
            or receipt_schema.get("contract_active_source_records_count") != 6
            or receipt_schema.get("contract_design_record_excluded_to_avoid_self_hash_cycle") is not True
            or type(receipt_schema.get("authority_active_source_records_count")) is not int
            or receipt_schema.get("authority_active_source_records_count") != 7
            or receipt_schema.get("authority_design_record_included") is not True
            or receipt.get("input_snapshots_exactly_equal") is not True
            or receipt.get("input_pre_snapshot") != receipt.get("input_post_snapshot")
            or cbytes(receipt.get("authorization")) != cbytes(module.AUTHORIZATION)
            or cbytes(receipt.get("runtime_observation")) != cbytes(module.RUNTIME)
            or cbytes(receipt.get("execution_boundary")) != cbytes(module.EXECUTION_BOUNDARY)
            or receipt.get("source_closure_sha256") != csha(receipt.get("source_closure"))
            or any(cbytes(receipt.get(key)) != cbytes(contract[key]) for key in module.EMBEDDED_FIELDS)
            or receipt.get("authority_design_contract") != regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES))
            or tree["inventory"] != [["authority_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]):
        raise RuntimeError("authority receipt")
    for row in receipt["source_closure"].values():
        if regular(Path(row["path"]), (row["sha256"], row["logical_bytes"])) != row:
            raise RuntimeError("authority source current")
    validate_authority_aliases(receipt, module)
    authorization = receipt.get("authorization", {})
    if (type(authorization.get("public_s1_zero_update_boundary_invocations_authorized")) is not int
            or authorization.get("public_s1_zero_update_boundary_invocations_authorized") != 1
            or type(authorization.get("public_s1_zero_update_boundary_invocations_consumed")) is not int
            or authorization.get("public_s1_zero_update_boundary_invocations_consumed") != 0
            or type(authorization.get("public_s1_evaluator_invocations_authorized")) is not int
            or authorization.get("public_s1_evaluator_invocations_authorized") != 1
            or type(authorization.get("zero_update_gate_invocations_authorized")) is not int
            or authorization.get("zero_update_gate_invocations_authorized") != 1
            or authorization.get("training_authorized") is not False
            or authorization.get("reward_read_authorized") is not False
            or authorization.get("hidden_private_or_final_input_authorized") is not False
            or authorization.get("retry_authorized") is not False):
        raise RuntimeError("public-S1 zero-update authority partition")
    durable = receipt.get("v534_actual_oof_durable_transition", {})
    if (durable.get("exact_count") != 11
            or type(durable.get("durable_partition", {}).get("actual_oof_execution_boundary_invocations")) is not int
            or durable.get("durable_partition", {}).get("actual_oof_execution_boundary_invocations") != 1
            or durable.get("durable_partition", {}).get("retry_authorized") is not False
            or type(durable.get("old_authority_receipt", {}).get("consumed")) is not int
            or durable.get("old_authority_receipt", {}).get("consumed") != 0):
        raise RuntimeError("v534 durable actual-OOF transition")
    failed = receipt.get("v539_poststage_owned_output_prep_failure_transition", {})
    if (failed.get("failure_forensic") != {"path": str(FORENSIC), "sha256": FORENSIC_SHA, "logical_bytes": FORENSIC_BYTES}
            or type(failed.get("old_authority_receipt", {}).get("consumed")) is not int
            or failed.get("old_authority_receipt", {}).get("consumed") != 0
            or failed.get("retry_authorized") is not False
            or failed.get("staged_exact8_before_failure") is not True):
        raise RuntimeError("v539 failed lineage forbidden reuse")
    for row in receipt.get("required_absences", {}).values():
        if row.get("absent") is not True or os.path.lexists(row.get("path", "")):
            raise RuntimeError("downstream absence")
    for path in (module.ATTEMPT_ROOT, module.ATTEMPT_PREP, module.OUTPUT_ROOT, module.OUTPUT_PREP):
        if os.path.lexists(path):
            raise RuntimeError("public-S1 downstream current")
    stdout_lines, native_record = transport_stdout_exact2(
        stdout_stream, stdout_path, stdout_identity, receipt_record, tree, writer)
    return {"authority_receipt": receipt_record, "authority_registration_tree": tree,
            "materializer_native_stdout": native_record,
            "stdout_line_origin": "transport_helper",
            "transport_stdout_lines": stdout_lines,
            "v534_actual_oof_durable_boundary_invocations": 1}


def base_receipt(status: str, passed: bool, invocations: int, cleanup: dict, error: BaseException | None = None) -> dict:
    row = {
        "format": "strict-track2-v540-v539-public-s1-zero-update-runtime-package-repair-authority-materializer-process-receipt-v1",
        "status": status, "passed": passed, "authority_materializer_invocations": invocations,
        "transport_helper_invocations": 1,
        "public_s1_zero_update_boundary_invocations": 0,
        "public_s1_evaluator_invocations": 0, "zero_update_gate_invocations": 0,
        "public_s1_independent_auditor_invocations": 0,
        "v534_actual_oof_durable_boundary_invocations": 1,
        "actual_oof_execution_invocations": 0,
        "failed_v539_transport_helper_normal_entries_non_durable": 1,
        "failed_v539_transport_helper_popen_invocations": 0,
        "phase_a_launcher_invocations": 0, "phase_a_driver_invocations": 0,
        "phase_a_worker_invocations": 0, "rng_proxy_delegate_invocations": 0,
        "oof_model_execution_invocations": 0, "phase_a_replay_invocations": 0,
        "training_invocations": 0, "cache_reuse_invocations": 0,
        "reward_read_invocations": 0, "dev_hidden_final_outcome_read_invocations": 0,
        "submission_invocations": 0,
        "retry_authorized": False, "cleanup": cleanup,
        "deployment_plan": deployment_plan(),
        "terminal_publication_contract": {
            "noreplace": True,
            "commit_semantics": "current_exact6_visibility_after_owned_temp_unlink",
            "crash_durability_claimed": False,
            "postcommit_parent_dir_fsync_best_effort": True,
            "postcommit_diagnostic_emitted_in_outer_transport_summary": True,
            "committed_process_receipt_never_mutated_for_diagnostics": True,
            "future_consumer_independent_current_exact6_gate_required": True,
            "no_rollback_relink_or_second_publish": True,
        },
    }
    if error is not None:
        row.update({"error_type": type(error).__name__, "error": str(error)})
    return row


def commit_terminal(root: Path, receipt: dict, baseline_mask, state: dict, phase_hook) -> dict:
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    root_stat = os.lstat(root)
    identity = (root_stat.st_dev, root_stat.st_ino)
    pre = exact_tree(root)
    if pre["file_count"] != 5 or [row[0] for row in pre["inventory"]] != EXACT5:
        raise RuntimeError("terminal precommit exact5")
    if not records_current(receipt):
        raise RuntimeError("terminal precommit records current")
    phase_hook("before_terminal")
    terminal_payload = cbytes(receipt)
    expected_terminal = {"path": str(root / "process_receipt.json"),
                         "sha256": hashlib.sha256(terminal_payload).hexdigest(),
                         "logical_bytes": len(terminal_payload)}
    expected_inventory = sorted(pre["inventory"] + [["process_receipt.json", expected_terminal["sha256"], expected_terminal["logical_bytes"]]])
    expected_lines = "".join(f"{digest}  {name}\n" for name, digest, _ in expected_inventory).encode()
    expected_tree = {"root": str(root), "inventory": expected_inventory, "file_count": 6,
                     "logical_file_bytes": sum(row[2] for row in expected_inventory),
                     "sha256sum_lines_digest_sha256": hashlib.sha256(expected_lines).hexdigest(),
                     "canonical_json_triples_digest_sha256": csha(expected_inventory)}
    def visible():
        state["visible"] = True
        state["receipt"] = receipt
        state["terminal_record"] = regular(root / "process_receipt.json")
    def committed():
        state["committed"] = True
        state["tree"] = expected_tree
        state["tree_observed_after_commit"] = False
    diagnostic = atomic_json_noreplace(root / "process_receipt.json", receipt, visible, committed)
    if diagnostic is not None:
        state["postcommit_diagnostic"] = diagnostic
    try:
        observed = os.lstat(root)
        tree = exact_tree(root)
        if ((observed.st_dev, observed.st_ino) != identity or tree != expected_tree
                or [row[0] for row in tree["inventory"]] != EXACT6
                or os.path.lexists(root / ".process_receipt.json.noreplace-tmp")
                or state["terminal_record"] != expected_terminal
                or not records_current(receipt)):
            raise RuntimeError("terminal exact6")
        state["tree"] = tree
        state["tree_observed_after_commit"] = True
        phase_hook("terminal_visible")
        phase_hook("terminal_committed")
    except BaseException as error:
        if not state["committed"]:
            raise
        state["postcommit_diagnostic"] = {"stage": "integrated_terminal_postcommit", "error_type": type(error).__name__, "error": str(error)}
    return state["tree"]


def orchestrate(*, root: Path, prep: Path, child_argv: list[str], helper_source: Path,
                 helper_record: dict, script_record: dict, argv_payload: dict,
                 timeout: float, success_validator, phase_hook=None,
                 retain_terminal_priority: bool = False,
                 failure_forensic_record: dict | None = None) -> dict:
    phase_hook = phase_hook or (lambda _phase: None)
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("evidence prestate")
    prep_identity = None
    prep_members = {}
    prep_owned = False
    promoted = False
    process = None
    stdout = None
    stderr = None
    stdout_identity = None
    stderr_identity = None
    stdout_flags = None
    invocations = 0
    state = {"visible": False, "committed": False, "authority_visible": False,
             "tree": None, "receipt": None,
             "terminal_record": None, "postcommit_diagnostic": None,
             "tree_observed_after_commit": False}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, _frame):
        if state["committed"] or state["authority_visible"]:
            return
        raise ControlledSignal(signum)
    for sig in old_handlers:
        signal.signal(sig, interrupted)
    baseline_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        prep.mkdir(mode=0o700)
        observed = os.lstat(prep)
        prep_identity = (observed.st_dev, observed.st_ino)
        prep_owned = True
        fsync_dir(prep.parent)
        phase_hook("prep_owned")
        write_exclusive(prep / "transport_helper.py", helper_source.read_bytes())
        regular(prep / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"]))
        prep_members["transport_helper.py"] = owned_member(prep / "transport_helper.py")
        serialized_argv = {key: value for key, value in argv_payload.items() if key != "pre_snapshot_builder"}
        write_exclusive(prep / "argv.json", cbytes({**serialized_argv, "transport_script": script_record}))
        prep_members["argv.json"] = owned_member(prep / "argv.json")
        stdout_descriptor = os.open(prep / "materializer_stdout.log",
                                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        stdout_stat = os.fstat(stdout_descriptor)
        stdout_identity = (stdout_stat.st_dev, stdout_stat.st_ino)
        observed_flags = fcntl.fcntl(stdout_descriptor, fcntl.F_GETFL)
        if (not stat.S_ISREG(stdout_stat.st_mode) or stdout_stat.st_size != 0
                or observed_flags & os.O_ACCMODE != os.O_RDWR):
            os.close(stdout_descriptor)
            raise RuntimeError("stdout empty O_RDWR regular")
        stdout_flags = {"open_flags": observed_flags, "access_mode": "O_RDWR",
                        "empty_regular_before_child": True, "held_from_before_popen_through_child_wait": True}
        os.fsync(stdout_descriptor)
        prep_members["materializer_stdout.log"] = owned_member(prep / "materializer_stdout.log")
        stdout = os.fdopen(stdout_descriptor, "r+b", buffering=0)
        stderr_descriptor = os.open(prep / "materializer_stderr.log",
                                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        stderr_stat = os.fstat(stderr_descriptor)
        stderr_identity = (stderr_stat.st_dev, stderr_stat.st_ino)
        if not stat.S_ISREG(stderr_stat.st_mode) or stderr_stat.st_size != 0:
            os.close(stderr_descriptor)
            raise RuntimeError("stderr empty O_RDWR regular")
        os.fsync(stderr_descriptor)
        stderr = os.fdopen(stderr_descriptor, "r+b", buffering=0)
        prep_members["materializer_stderr.log"] = owned_member(prep / "materializer_stderr.log")
        promoted_argv_record = regular(prep / "argv.json")
        promoted_argv_record["path"] = str(root / "argv.json")
        write_exclusive(prep / "intent.json", cbytes({
            "format": "strict-track2-v540-v539-public-s1-zero-update-runtime-package-repair-authority-materializer-intent-v1",
            "status": "committed_before_exact_once_authority_materializer", "retry_authorized": False,
            "argv": promoted_argv_record,
        }))
        prep_members["intent.json"] = owned_member(prep / "intent.json")
        fsync_dir(prep)
        if [row[0] for row in exact_tree(prep)["inventory"]] != EXACT5:
            raise RuntimeError("prep exact5")
        assert_owned_fd(stdout, prep / "materializer_stdout.log", stdout_identity)
        assert_owned_fd(stderr, prep / "materializer_stderr.log", stderr_identity)
        phase_hook("before_promote")
        rename_directory_noreplace(prep, root)
        fsync_dir(root.parent)
        promoted_stat = os.lstat(root)
        if ((promoted_stat.st_dev, promoted_stat.st_ino) != prep_identity
                or root.is_symlink() or not root.is_dir()
                or [row[0] for row in exact_tree(root)["inventory"]] != EXACT5):
            raise RuntimeError("evidence promoted ownership/exact5")
        prep_owned = False
        promoted = True
        # Rebuild after promote: this exact record must name the final persistent path.
        helper_copy = regular(root / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"]))
        phase_hook("after_promote_before_unmask")
        assert_owned_fd(stdout, root / "materializer_stdout.log", stdout_identity)
        assert_owned_fd(stderr, root / "materializer_stderr.log", stderr_identity)
        if regular(root / "materializer_stdout.log")["logical_bytes"] != 0:
            raise RuntimeError("stdout prechild empty")
        spawn_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        phase_hook("before_popen")
        process = subprocess.Popen(
            child_argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
            preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask))
        invocations = 1
        phase_hook("popen_owned")
        returncode = process.wait(timeout=timeout)
        fsync_stream(stdout)
        fsync_stream(stderr)
        assert_owned_fd(stdout, root / "materializer_stdout.log", stdout_identity)
        assert_owned_fd(stderr, root / "materializer_stderr.log", stderr_identity)
        cleanup = terminate(process)
        extra = success_validator(returncode, cleanup, stdout, stdout_identity)
        close_fsync(stdout); stdout = None
        close_fsync(stderr); stderr = None
        phase_hook("authority_exact1_before_flag")
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        state["authority_visible"] = True
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        phase_hook("after_authority_before_terminal")
        post_snapshot = argv_payload["pre_snapshot_builder"]()
        if post_snapshot != argv_payload["pre_snapshot"]:
            raise RuntimeError("immutable snapshot drift")
        receipt = {
            **base_receipt("passed_exact_once_public_s1_zero_update_authority_materialized_after_absence_schema_repair_no_gate_execution", True, 1, cleanup),
            "helper_returncode": 0, "materializer_returncode": returncode,
            "transport_helper": helper_record, "transport_helper_copy": helper_copy,
            "transport_script": script_record, "argv": regular(root / "argv.json"),
            "transport_deployment_record": argv_payload["transport_deployment_record"],
            "intent": regular(root / "intent.json"), "stdout": regular(root / "materializer_stdout.log"),
            "stderr": regular(root / "materializer_stderr.log"),
            "pre_snapshot": argv_payload["pre_snapshot"], "post_snapshot": post_snapshot,
            "pre_post_snapshots_exactly_equal": True, "stdout_descriptor_evidence": stdout_flags, **extra,
        }
        if failure_forensic_record is not None:
            receipt["failed_transport_forensic"] = failure_forensic_record
        # Builder callable is transport-local and never serialized.
        receipt["argv_payload_sha256"] = csha({key: value for key, value in argv_payload.items() if key != "pre_snapshot_builder"})
        tree = commit_terminal(root, receipt, baseline_mask, state, phase_hook)
        return {"passed": True, "process_receipt": receipt, "evidence_tree": tree,
                "evidence_tree_observed_after_commit": state["tree_observed_after_commit"],
                "postcommit_diagnostic": state["postcommit_diagnostic"]}
    except BaseException as error:
        if state["committed"]:
            committed_receipt = state["receipt"]
            committed_tree = state["tree"]
            state["postcommit_diagnostic"] = {"stage": "orchestrate_committed_handler", "error_type": type(error).__name__, "error": str(error)}
            return {"passed": committed_receipt.get("passed") is True,
                    "process_receipt": committed_receipt, "evidence_tree": committed_tree,
                    "evidence_tree_observed_after_commit": state["tree_observed_after_commit"],
                    "postcommit_diagnostic": state["postcommit_diagnostic"]}
        if state["visible"]:
            # A no-replace terminal link is already externally visible, but
            # unlink/fsync/exact6 did not finish.  Preserve that first terminal,
            # never attempt a second publish, and force a nonzero transport exit.
            raise
        if promoted:
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try: close_fsync(stdout)
            except BaseException: pass
            try: close_fsync(stderr)
            except BaseException: pass
            stdout = None
            stderr = None
            cleanup = terminate(process)
            for log_name, expected_identity in (
                    ("materializer_stdout.log", stdout_identity),
                    ("materializer_stderr.log", stderr_identity)):
                metadata = os.lstat(root / log_name)
                if expected_identity is None or (metadata.st_dev, metadata.st_ino) != expected_identity:
                    raise RuntimeError("owned log identity drift; refusing terminal") from error
            if not state["committed"]:
                failure = {
                    **base_receipt("failed_no_retry", False, invocations, cleanup, error),
                    "transport_helper": helper_record,
                    "transport_helper_copy": regular(root / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"])),
                    "transport_script": script_record,
                    "transport_deployment_record": argv_payload["transport_deployment_record"],
                    "stdout_descriptor_evidence": stdout_flags,
                    "argv": regular(root / "argv.json"), "intent": regular(root / "intent.json"),
                    "stdout": regular(root / "materializer_stdout.log"), "stderr": regular(root / "materializer_stderr.log"),
                }
                if failure_forensic_record is not None:
                    failure["failed_transport_forensic"] = failure_forensic_record
                tree = commit_terminal(root, failure, baseline_mask, state, phase_hook)
                return {"passed": False, "process_receipt": failure, "evidence_tree": tree, "error": error}
        if prep_owned:
            try: close_fsync(stdout)
            except BaseException: pass
            try: close_fsync(stderr)
            except BaseException: pass
            cleanup_owned_prep(prep, prep_identity, prep_members)
        raise
    finally:
        if state["committed"] and retain_terminal_priority:
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try: signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
            except BaseException: pass
        else:
            try: signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
            except BaseException: pass
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def production_fixture(contract_source: Path = CONTRACT, materializer_source: Path = MATERIALIZER,
                       module_overrides: dict | None = None) -> dict:
    module, contract, contract_record, materializer_record = load_materializer(
        contract_source, materializer_source, module_overrides)
    source = Path(__file__).read_text(encoding="utf-8")
    syntax = ast.parse(source)
    popens = [node for node in ast.walk(syntax) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    plan = deployment_plan()
    checks = {
        "actual_materializer_contract_consumer": (
            contract_record["sha256"] == CONTRACT_SHA and materializer_record["sha256"] == MATERIALIZER_SHA),
        "final8_hardbound": all(value in source for value in FROZEN_FINAL8_SHA256),
        "fresh_final8_transport3_deploy11": (
            len(plan["fresh_final8_noreplace"]) == 8
            and len(plan["fresh_final8_volatile_sources"]) == 8
            and plan["fresh_noncyclic_transport_triple_external_noreplace_required"] is True
            and plan["fresh_transport_invocations_before_normal"] == 0
            and plan["atomic_source_publication_contract"]["manifest_large_file_bytes"] == MANIFEST_BYTES
            and plan["atomic_source_publication_contract"]["held_descriptor_through_link_and_owned_unlink"] is True
            and plan["atomic_source_publication_contract"]["pread_exact_sha_bytes_and_eof_required"] is True),
        "absence_schema_exact5_bijection_hash_source": (
            EXPECTED_CURRENT_ABSENCE_MAP_SHA256 in source
            and EXPECTED_CURRENT_ABSENCE_KEYS == ["attempt_prep", "attempt_root", "authority_prep", "output_prep", "output_root"]
            and set(EXPECTED_CURRENT_ABSENCE_PATHS) == set(EXPECTED_CURRENT_ABSENCE_KEYS)),
        "no_dynamic_execution": all(getattr(node, "func", None) is None
                    or not isinstance(getattr(node, "func", None), ast.Name)
                    or getattr(node.func, "id", "") not in {"exec", "eval"}
                    for node in ast.walk(syntax)),
        "sole_signal_owned_materializer_popen": len(popens) == 1 and command()[1] == str(MATERIALIZER),
        "authority_schema_69_31_35": (
            len(contract["authority_receipt_contract"]["top_keys"]) == 69
            and len(module.CHECK_KEYS) == 31 and len(module.AUTHORITY_SOURCE_ROLES) == 35),
        "public_s1_authorization_exact": (
            module.AUTHORIZATION["public_s1_zero_update_boundary_invocations_authorized"] == 1
            and module.AUTHORIZATION["public_s1_zero_update_boundary_invocations_consumed"] == 0
            and module.AUTHORIZATION["public_s1_evaluator_invocations_authorized"] == 1
            and module.AUTHORIZATION["zero_update_gate_invocations_authorized"] == 1
            and module.AUTHORIZATION["training_authorized"] is False
            and module.AUTHORIZATION["hidden_private_or_final_input_authorized"] is False),
        "downstream_not_in_child_argv": (
            all(str(path) not in command() for path in (EVALUATOR, AUDITOR, LAUNCHER))
            and not any("launch_" in item or "evaluate_" in item or "audit_" in item for item in command())),
        "native_exact1_to_transport_exact2_source": (
            "materializer native stdout exact1" in source and "transport stdout exact2 readback" in source),
        "choice_a_external_evidence_exact6": EXACT6 == sorted(EXACT5 + ["process_receipt.json"]),
    }
    with tempfile.TemporaryDirectory(prefix="v540-stage-a-helper-", dir="/dev/shm") as folder:
        temp = Path(folder)
        script = temp / "transport.sh"
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        helper_record = regular(Path(__file__))
        script_record = regular(script)
        deployment = temp / "transport-deployment.json"
        deployment.write_bytes(cbytes(expected_transport_deployment(helper_record, script_record)))
        deployment_record = regular(deployment)
        pre = {"stable": True}
        root = temp / "normal"
        result = orchestrate(
            root=root, prep=temp / "normal.execution-prep",
            child_argv=[str(RLPY), "-c", "pass"], helper_source=Path(__file__),
            helper_record=helper_record, script_record=script_record,
            argv_payload={"format": "synthetic-v540-stage-a", "pre_snapshot": pre,
                          "transport_deployment_record": deployment_record,
                          "pre_snapshot_builder": lambda: pre}, timeout=30,
            success_validator=lambda rc, cleanup, stream, identity: {
                "synthetic_returncode": rc, "synthetic_cleanup": cleanup},
            retain_terminal_priority=True)
        checks["mapped_normal_exact6"] = (
            result["passed"] is True and exact_tree(root)["file_count"] == 6
            and result["process_receipt"]["public_s1_zero_update_boundary_invocations"] == 0)
        postcommit_root = temp / "integrated_postcommit_fault"
        original_exact_tree = globals()["exact_tree"]
        integrated_tree_fault = {"injected": False}
        def integrated_postcommit_exact_tree(path):
            if (path == postcommit_root
                    and os.path.lexists(path / "process_receipt.json")
                    and not integrated_tree_fault["injected"]):
                integrated_tree_fault["injected"] = True
                raise RuntimeError("integrated postcommit exact-tree diagnostic")
            return original_exact_tree(path)
        globals()["exact_tree"] = integrated_postcommit_exact_tree
        try:
            postcommit = orchestrate(
                root=postcommit_root, prep=temp / "integrated_postcommit_fault.execution-prep",
                child_argv=[str(RLPY), "-c", "pass"], helper_source=Path(__file__),
                helper_record=helper_record, script_record=script_record,
                argv_payload={"format": "synthetic-v540-stage-a", "pre_snapshot": pre,
                              "transport_deployment_record": deployment_record,
                              "pre_snapshot_builder": lambda: pre}, timeout=30,
                success_validator=lambda rc, cleanup, stream, identity: {
                    "synthetic_returncode": rc, "synthetic_cleanup": cleanup},
                retain_terminal_priority=True)
        finally:
            globals()["exact_tree"] = original_exact_tree
        independently_observed_postcommit_tree = exact_tree(postcommit_root)
        checks["integrated_postcommit_fault_success_priority_exact6"] = (
            integrated_tree_fault["injected"] is True
            and postcommit["passed"] is True
            and postcommit["postcommit_diagnostic"]["stage"] == "integrated_terminal_postcommit"
            and postcommit["evidence_tree_observed_after_commit"] is False
            and postcommit["evidence_tree"] == independently_observed_postcommit_tree
            and independently_observed_postcommit_tree["file_count"] == 6
            and [row[0] for row in independently_observed_postcommit_tree["inventory"]] == EXACT6)
        fail_root = temp / "nonzero"
        failed = orchestrate(
            root=fail_root, prep=temp / "nonzero.execution-prep",
            child_argv=[str(RLPY), "-c", "raise SystemExit(9)"], helper_source=Path(__file__),
            helper_record=helper_record, script_record=script_record,
            argv_payload={"format": "synthetic-v540-stage-a", "pre_snapshot": pre,
                          "transport_deployment_record": deployment_record,
                          "pre_snapshot_builder": lambda: pre}, timeout=30,
            success_validator=lambda rc, cleanup, stream, identity: (_ for _ in ()).throw(RuntimeError(rc)),
            retain_terminal_priority=True)
        checks["mapped_nonzero_failed_exact6"] = (
            failed["passed"] is False and exact_tree(fail_root)["file_count"] == 6
            and failed["process_receipt"]["status"] == "failed_no_retry")
        popen_root = temp / "popen_error"
        popen_failed = orchestrate(
            root=popen_root, prep=temp / "popen_error.execution-prep",
            child_argv=[str(temp / "missing-executable")], helper_source=Path(__file__),
            helper_record=helper_record, script_record=script_record,
            argv_payload={"format": "synthetic-v540-stage-a", "pre_snapshot": pre,
                          "transport_deployment_record": deployment_record,
                          "pre_snapshot_builder": lambda: pre}, timeout=1,
            success_validator=lambda *_: {}, retain_terminal_priority=True)
        checks["mapped_popen_error_failed_exact6"] = (
            popen_failed["passed"] is False and exact_tree(popen_root)["file_count"] == 6
            and popen_failed["process_receipt"]["authority_materializer_invocations"] == 0)

        primitive = temp / "terminal-primitives"
        primitive.mkdir()
        visible = {"value": False}; committed = {"value": False}
        original_fsync = globals()["fsync_terminal_parent"]
        globals()["fsync_terminal_parent"] = lambda _path: (_ for _ in ()).throw(OSError("fsync fault"))
        try:
            target = primitive / "postcommit-fsync.json"
            atomic_json_noreplace(target, {"passed": True},
                                  lambda: visible.__setitem__("value", True),
                                  lambda: committed.__setitem__("value", True))
            checks["choice_a_postcommit_fsync_success_priority"] = (
                visible["value"] and committed["value"] and target.exists()
                and not target.with_name("." + target.name + ".noreplace-tmp").exists())
        finally:
            globals()["fsync_terminal_parent"] = original_fsync
        target = primitive / "pre-record-replacement.json"
        def replace_before_record(stage, path, _descriptor):
            if stage == "before_first_identity":
                path.unlink(); path.write_bytes(cbytes({"passed": True}))
        try:
            atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None,
                                  fixture_hook=replace_before_record)
        except RuntimeError:
            checks["terminal_pre_record_same_content_replacement_rejected_preserved"] = (
                not target.exists() and target.with_name("." + target.name + ".noreplace-tmp").read_bytes() == cbytes({"passed": True}))
        else:
            checks["terminal_pre_record_same_content_replacement_rejected_preserved"] = False
        target = primitive / "after-record-replacement.json"
        def replace_after_record(stage, path, _descriptor):
            if stage == "after_record":
                path.unlink(); path.write_text("foreign", encoding="utf-8")
        try:
            atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None,
                                  fixture_hook=replace_after_record)
        except RuntimeError:
            checks["terminal_after_record_replacement_rejected_preserved"] = (
                not target.exists() and target.with_name("." + target.name + ".noreplace-tmp").read_text() == "foreign")
        else:
            checks["terminal_after_record_replacement_rejected_preserved"] = False
        original_write = globals()["terminal_write"]
        globals()["terminal_write"] = lambda _descriptor, _payload: 0
        try:
            target = primitive / "short-write.json"
            try:
                atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None)
            except RuntimeError:
                checks["terminal_shortwrite_owned_cleanup"] = (
                    not target.exists() and not target.with_name("." + target.name + ".noreplace-tmp").exists())
            else:
                checks["terminal_shortwrite_owned_cleanup"] = False
        finally:
            globals()["terminal_write"] = original_write
        original_pread = globals()["terminal_pread"]
        eof_fault = {"injected": False}
        def one_shot_eof(descriptor, size, offset):
            actual = original_pread(descriptor, size, offset)
            if size == 1 and actual == b"" and not eof_fault["injected"]:
                eof_fault["injected"] = True
                return b"x"
            return actual
        globals()["terminal_pread"] = one_shot_eof
        try:
            target = primitive / "eof-fault.json"
            try:
                atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None)
            except RuntimeError:
                checks["terminal_eof_fault_owned_cleanup"] = (
                    eof_fault["injected"] and not target.exists()
                    and not target.with_name("." + target.name + ".noreplace-tmp").exists())
            else:
                checks["terminal_eof_fault_owned_cleanup"] = False
        finally:
            globals()["terminal_pread"] = original_pread
        target = primitive / "pending-signal.json"
        def pending_signal(stage, _path, _descriptor):
            if stage == "after_record":
                raise ControlledSignal(signal.SIGTERM)
        try:
            atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None,
                                  fixture_hook=pending_signal)
        except ControlledSignal:
            checks["terminal_pending_signal_owned_cleanup"] = (
                not target.exists() and not target.with_name("." + target.name + ".noreplace-tmp").exists())
        else:
            checks["terminal_pending_signal_owned_cleanup"] = False
        original_unlink = globals()["unlink_terminal_temp"]
        globals()["unlink_terminal_temp"] = lambda _path: (_ for _ in ()).throw(OSError("unlink fault"))
        try:
            target = primitive / "unlink-fault.json"
            try:
                atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None)
            except OSError:
                checks["terminal_unlink_fault_exact2_noncommitted_no_second"] = (
                    target.exists() and target.with_name("." + target.name + ".noreplace-tmp").exists())
            else:
                checks["terminal_unlink_fault_exact2_noncommitted_no_second"] = False
        finally:
            globals()["unlink_terminal_temp"] = original_unlink
        target = primitive / "postcommit-current-fault.json"
        def postcommit_current_fault(stage, path, _descriptor):
            if stage == "postcommit":
                target_path = path.with_name(path.name.removeprefix(".").removesuffix(".noreplace-tmp"))
                target_path.unlink()
                target_path.write_text("foreign-current", encoding="utf-8")
                raise RuntimeError("postcommit current diagnostic")
        atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None,
                              fixture_hook=postcommit_current_fault)
        checks["terminal_postcommit_current_fault_success_priority_no_second"] = (
            target.read_text() == "foreign-current"
            and not target.with_name("." + target.name + ".noreplace-tmp").exists())
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"passed": True, "checks": checks, "checks_sha256": csha(checks),
            "materializer_invocations": 0, "public_s1_zero_update_boundary_invocations": 0,
            "evaluator_invocations": 0, "auditor_invocations": 0, "launcher_invocations": 0,
            "oof_readonly_gate_validator_invocations": 0, "phase_a_invocations": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha")
    parser.add_argument("--self-bytes", type=int)
    parser.add_argument("--script-sha")
    parser.add_argument("--script-bytes", type=int)
    parser.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--read-only-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        print(json.dumps(production_fixture(), sort_keys=True)); return 0
    helper_record = regular(SELF_PATH, (args.self_sha, args.self_bytes))
    script_record = regular(SCRIPT_PATH, (args.script_sha, args.script_bytes))
    deployment_record = validate_transport_deployment_record(helper_record, script_record)
    module, contract, contract_record, materializer_record = load_materializer()
    pre = immutable_snapshot(contract, helper_record, script_record)
    if pre["gpu_compute_pids"] or pre["phase_a_or_reconciler_pids"]:
        raise RuntimeError("PID/GPU prestate")
    if any(os.path.lexists(path) for path in (
            AUTHORITY_ROOT, AUTHORITY_PREP, EVIDENCE_ROOT, EVIDENCE_PREP)):
        raise RuntimeError("runtime prestate")
    if args.read_only_preflight:
        print(json.dumps({"passed": True, "read_only_preflight": True,
                          "contract": contract_record, "materializer": materializer_record,
                          "stable_snapshot_sha256": csha(pre)}, sort_keys=True))
        return 0
    argv = command()
    payload = {
        "format": "strict-track2-v540-v539-public-s1-zero-update-runtime-package-repair-authority-materializer-argv-v1",
        "argv": argv, "argv_repr": repr(argv), "argv_utf8_hex": [item.encode().hex() for item in argv],
        "authority_contract": contract_record, "authority_materializer": materializer_record,
        "transport_helper": helper_record,
        "transport_deployment_record": deployment_record,
        "pre_snapshot": pre, "pre_snapshot_builder": lambda: immutable_snapshot(contract, helper_record, script_record),
    }
    result = orchestrate(root=EVIDENCE_ROOT, prep=EVIDENCE_PREP, child_argv=argv,
                         helper_source=SELF_PATH, helper_record=helper_record, script_record=script_record,
                         argv_payload=payload, timeout=300,
                         success_validator=lambda rc, clean, stream, identity: validate_success(
                             module, contract, rc, clean, stream, identity),
                         retain_terminal_priority=True)
    if not result["passed"]:
        raise RuntimeError(str(result["error"]))
    print(json.dumps({"passed": True, "evidence_tree": result["evidence_tree"],
                      "evidence_tree_observed_after_commit": result.get("evidence_tree_observed_after_commit"),
                      "postcommit_diagnostic": result.get("postcommit_diagnostic")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
