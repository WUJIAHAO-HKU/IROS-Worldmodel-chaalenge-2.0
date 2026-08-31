#!/usr/bin/env python3
"""Exact-once Stage-A transport for the v525 read-only reconciliation authority."""
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
SELF_PATH = SCRIPTS / "invoke_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_once.py"
SCRIPT_PATH = Path("/root/v525_phase_a_worker_receipt_format_readonly_reconciliation_once.sh")
CONTRACT = SCRIPTS / "v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_contract.json"
CONTRACT_SHA = "749373c708147ff7079c3efb2196a046a185edacccfca32a44a59d87e5d50124"
CONTRACT_BYTES = 70602
MATERIALIZER = SCRIPTS / "materialize_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority.py"
MATERIALIZER_SHA = "3b8434571c7444c489689c2b7601bb56e79a46d79780843875f4cc5b6fcf9e82"
MATERIALIZER_BYTES = 27325
RECONCILER = SCRIPTS / "reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py"
RECONCILER_SHA = "81aa7da1eb525b5d03f8053da4dbce81e372b0660b6aa31e068dc3e2604b13f3"
RECONCILER_BYTES = 48538
FAILURE_FORENSIC = SCRIPTS / "v525_v524_phase_a_worker_receipt_format_failure_forensic.json"
FAILURE_FORENSIC_SHA = "ea04aede510143cf9f1c6388c222b64397a8f5dfdf3c15a5c10c56a7d5b058d5"
FAILURE_FORENSIC_BYTES = 8137
AUTHORITY_ROOT = J / "v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_seed1661_20260827"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")
AUTHORITY_RECEIPT = AUTHORITY_ROOT / "authority_receipt.json"
AUTHORITY_RECEIPT_SHA = "1f468f2c5ed2f5928e9491e7113f1f34e1095dcd17c57c2a9c973107043a4057"
AUTHORITY_RECEIPT_BYTES = 95906
STAGE_A_EVIDENCE = J / "v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_materialization_evidence_seed1661_20260827"
STAGE_A_EVIDENCE_LINES = "d500f32ec9a5c60c086fb4eef9d98191d16e2e64311c3b731921ba165fe7aef6"
STAGE_A_EVIDENCE_CANON = "7c455785a8675f1214f793fd93355ee8145002d758e2eabe14bba29fb2da36b3"
STAGE_A_PROCESS_SHA = "fb30407693464e98feae18eb81ec52905a62a5a4d160889505e86ecaaa8a31f8"
STAGE_A_PROCESS_BYTES = 31038
CANDIDATE_ROOT = J / "v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_seed1661_20260827"
CANDIDATE_PREP = CANDIDATE_ROOT.with_name(CANDIDATE_ROOT.name + ".reconciliation-prep")
CANDIDATE_RECEIPT = CANDIDATE_ROOT / "reconciliation_receipt.json"
EVIDENCE_ROOT = J / "v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_attempt_seed1661_20260827"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".attempt-prep")
V524_QUALIFICATION = Path("/root/v524_v523_phase_a_cache_qualification_seed1660_20260826")
EXACT5 = ["argv.json", "intent.json", "reconciler_stderr.log", "reconciler_stdout.log", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])
AT_FDCWD = -100
RENAME_NOREPLACE = 1
FROZEN_FINAL4_SHA256 = (CONTRACT_SHA, MATERIALIZER_SHA, RECONCILER_SHA, FAILURE_FORENSIC_SHA)
OLD_FORMAT = "strict-track2-v523-v522-rng-isolated-cache-worker-receipt-v1"
NORMALIZED_FORMAT = "strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1"
RAW_MEDIAN_WARNING = "median CUDA with indices output does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)"


def deployment_plan() -> dict:
    return {
        "stage_a_final4_current_exact": [
            {"role": "authority_design_contract", "path": str(CONTRACT), "sha256": CONTRACT_SHA, "logical_bytes": CONTRACT_BYTES},
            {"role": "authority_materializer", "path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
            {"role": "readonly_reconciler", "path": str(RECONCILER), "sha256": RECONCILER_SHA, "logical_bytes": RECONCILER_BYTES},
            {"role": "failure_forensic", "path": str(FAILURE_FORENSIC), "sha256": FAILURE_FORENSIC_SHA, "logical_bytes": FAILURE_FORENSIC_BYTES}],
        "stage_a_authority_receipt": {"path": str(AUTHORITY_RECEIPT), "sha256": AUTHORITY_RECEIPT_SHA, "logical_bytes": AUTHORITY_RECEIPT_BYTES},
        "stage_a_evidence_exact6_required": True,
        "stage_b_fresh_transport_sources_noreplace": [str(SELF_PATH), str(SCRIPT_PATH)],
        "canonical_readback_exact_required": True,
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


def unlink_terminal_temp(path: Path) -> None:
    path.unlink()


def fsync_terminal_parent(path: Path) -> None:
    fsync_dir(path)


def atomic_json_noreplace(path: Path, value: object, visible_hook, committed_hook, fixture_hook=None) -> None:
    temporary = path.with_name("." + path.name + ".noreplace-tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise FileExistsError(path)
    write_exclusive(temporary, cbytes(value))
    temporary_metadata = os.lstat(temporary)
    temporary_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
    temporary_record = regular(temporary)
    linked = False
    try:
        if fixture_hook is not None:
            fixture_hook(temporary)
        current_metadata = os.lstat(temporary)
        if ((current_metadata.st_dev, current_metadata.st_ino) != temporary_identity
                or regular(temporary) != temporary_record):
            raise RuntimeError("terminal temp ownership drift")
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        visible_hook()
        unlink_terminal_temp(temporary)
        committed_hook()
        try:
            fsync_terminal_parent(path.parent)
        except BaseException:
            # Choice-A visibility commit: after the owned temp unlink syscall
            # succeeds, parent fsync is best-effort and cannot revoke success.
            pass
    except BaseException:
        if not linked and os.path.lexists(temporary):
            current_metadata = os.lstat(temporary)
            if (temporary.is_symlink() or not stat.S_ISREG(current_metadata.st_mode)
                    or (current_metadata.st_dev, current_metadata.st_ino) != temporary_identity
                    or regular(temporary) != temporary_record):
                raise RuntimeError("terminal temp cleanup ownership drift")
            temporary.unlink()
            fsync_dir(path.parent)
        raise


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


def phase_a_or_reconciler_pids() -> list[dict]:
    markers = ("launch_v524_v523_phase_a_cache_qualification.py",
               "generate_v524_v523_phase_a_cache_qualification.py",
               "generate_v524_v523_phase_a_cache_qualification_worker.py",
               "reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py")
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(marker in command for marker in markers):
            rows.append({"pid": int(entry.name), "cmdline": command})
    return sorted(rows, key=lambda row: row["pid"])


def stable_absences(contract: dict) -> dict:
    absences = {name: {"path": str(path), "absent": not os.path.lexists(path)} for name, path in {
        "candidate_root": CANDIDATE_ROOT, "candidate_prep": CANDIDATE_PREP,
        "external_evidence_root": EVIDENCE_ROOT, "external_evidence_prep": EVIDENCE_PREP}.items()}
    if not all(row["absent"] for row in absences.values()):
        raise RuntimeError("contract current absence")
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
    # Dynamic Stage-B roots are verified once before launch and independently
    # after the child; they are intentionally excluded from stable equality.
    absences = {}
    authority_record = regular(AUTHORITY_RECEIPT, (AUTHORITY_RECEIPT_SHA, AUTHORITY_RECEIPT_BYTES))
    authority_tree = exact_tree(AUTHORITY_ROOT)
    if authority_tree["file_count"] != 1 or authority_tree["inventory"] != [["authority_receipt.json", AUTHORITY_RECEIPT_SHA, AUTHORITY_RECEIPT_BYTES]]:
        raise RuntimeError("stage-A authority exact1")
    stage_a_tree = exact_tree(STAGE_A_EVIDENCE)
    if (stage_a_tree["file_count"] != 6 or stage_a_tree["sha256sum_lines_digest_sha256"] != STAGE_A_EVIDENCE_LINES
            or stage_a_tree["canonical_json_triples_digest_sha256"] != STAGE_A_EVIDENCE_CANON
            or regular(STAGE_A_EVIDENCE / "process_receipt.json", (STAGE_A_PROCESS_SHA, STAGE_A_PROCESS_BYTES)) is None):
        raise RuntimeError("stage-A evidence exact6")
    stage_a_process = json.loads((STAGE_A_EVIDENCE / "process_receipt.json").read_text())
    if stage_a_process.get("passed") is not True or stage_a_process.get("status") != "passed_exact_once_authority_materialized_no_phase_a_execution":
        raise RuntimeError("stage-A process")
    return {
        "contract": regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES)),
        "materializer": regular(MATERIALIZER, (MATERIALIZER_SHA, MATERIALIZER_BYTES)),
        "transport_helper": regular(SELF_PATH, (helper_record["sha256"], helper_record["logical_bytes"])),
        "transport_script": regular(SCRIPT_PATH, (script_record["sha256"], script_record["logical_bytes"])),
        "sources": sources, "trees": trees, "absences": absences,
        "stage_a_authority_receipt": authority_record, "stage_a_authority_tree": authority_tree,
        "stage_a_evidence_tree": stage_a_tree,
        "service_health": service_health(), "gpu_compute_pids": gpu_compute_pids(),
        "phase_a_or_reconciler_pids": phase_a_or_reconciler_pids(), "deployment_plan": deployment_plan(),
    }


def validate_materializer_contract(module, contract: dict, materializer_record: dict) -> None:
    seed = contract.get("seed")
    active = {
        "authority_materializer": {"path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
        "readonly_reconciler": {"path": str(RECONCILER), "sha256": RECONCILER_SHA, "logical_bytes": RECONCILER_BYTES},
        "failure_forensic": {"path": str(FAILURE_FORENSIC), "sha256": FAILURE_FORENSIC_SHA, "logical_bytes": FAILURE_FORENSIC_BYTES},
    }
    if (module.CONTRACT_PATH != CONTRACT or module.AUTHORITY_ROOT != AUTHORITY_ROOT
            or set(contract) != module.CONTRACT_TOP_KEYS
            or contract.get("format") != module.CONTRACT_FORMAT
            or contract.get("status") != module.CONTRACT_STATUS
            or type(seed) is not int or seed != 1661
            or contract.get("authority_materializer_source") != materializer_record
            or any(contract.get("source_closure", {}).get(role) != row for role, row in active.items())
            or len(contract.get("authority_receipt_contract", {}).get("top_keys", [])) != 55
            or len(module.CHECK_KEYS) != 31
            or len(module.CONTRACT_SOURCE_ORDER) != 16 or len(module.AUTHORITY_SOURCE_ORDER) != 17
            or contract.get("source_role_order") != module.CONTRACT_SOURCE_ORDER
            or contract.get("source_aliases") != {role: module.SOURCE_ALIASES[role] for role in module.CONTRACT_SOURCE_ORDER}):
        raise RuntimeError("materializer/contract schema")
    if (cbytes(contract.get("reconciler_candidate_contract")) != cbytes(module.RECONCILER_CANDIDATE_CONTRACT)
            or cbytes(contract.get("external_terminal_evidence_contract")) != cbytes(module.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT)
            or cbytes(contract.get("authorization")) != cbytes(module.AUTHORIZATION)
            or csha(contract.get("runtime_observation")) != csha(module.RUNTIME)
            or csha(contract.get("execution_boundary")) != csha(module.EXECUTION_BOUNDARY)):
        raise RuntimeError("v525 candidate/external-terminal contract")
    for role, row in contract["source_closure"].items():
        if regular(Path(row["path"]), (row["sha256"], row["logical_bytes"])) != row:
            raise RuntimeError(f"source closure current: {role}")
    module.validate_state(contract)


def load_materializer(contract_source: Path = CONTRACT, materializer_source: Path = MATERIALIZER,
                      module_overrides: dict | None = None):
    contract_record = regular(contract_source, (CONTRACT_SHA, CONTRACT_BYTES))
    materializer_record = regular(materializer_source, (MATERIALIZER_SHA, MATERIALIZER_BYTES))
    if contract_source != CONTRACT:
        contract_record = {**contract_record, "path": str(CONTRACT)}
    if materializer_source != MATERIALIZER:
        materializer_record = {**materializer_record, "path": str(MATERIALIZER)}
    contract = json.loads(contract_source.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("v525_materializer", materializer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in (module_overrides or {}).items():
        setattr(module, name, value)
    validate_materializer_contract(module, contract, materializer_record)
    return module, contract, contract_record, materializer_record


def command() -> list[str]:
    return [str(RLPY), str(RECONCILER),
            "--authority-contract", str(CONTRACT), "--authority-contract-sha", CONTRACT_SHA,
            "--authority-contract-bytes", str(CONTRACT_BYTES),
            "--authority-receipt", str(AUTHORITY_RECEIPT), "--authority-receipt-sha", AUTHORITY_RECEIPT_SHA,
            "--authority-receipt-bytes", str(AUTHORITY_RECEIPT_BYTES),
            "--authority-materializer-source", str(MATERIALIZER), "--authority-materializer-sha", MATERIALIZER_SHA,
            "--authority-materializer-bytes", str(MATERIALIZER_BYTES),
            "--reconciler-source", str(RECONCILER), "--reconciler-sha", RECONCILER_SHA,
            "--reconciler-bytes", str(RECONCILER_BYTES),
            "--forensic", str(FAILURE_FORENSIC), "--forensic-sha", FAILURE_FORENSIC_SHA,
            "--forensic-bytes", str(FAILURE_FORENSIC_BYTES)]


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
                            candidate_record: dict, candidate_tree: dict,
                            diagnostic: dict, writer=os.write) -> tuple[list[dict], dict]:
    assert_owned_fd(stream, path, identity)
    descriptor = stream.fileno()
    try:
        native = read_fd_all_exact(descriptor)
        native_record = {
            "line_origin": "reconciler_native_capture", "logical_bytes": len(native),
            "sha256": hashlib.sha256(native).hexdigest(), "eof_exact": True,
        }
        lines = [
            {"line_origin": "transport_helper", "reconciler_native_capture": native_record,
             "candidate_receipt": candidate_record, "candidate_tree": candidate_tree},
            {"line_origin": "transport_helper", "passed": True,
              "transport_helper_invocations": 1, "authority_materializer_invocations": 0,
              "readonly_reconciler_invocations": 1, "child_diagnostic": diagnostic,
             "phase_a_launcher_invocations": 0,
             "phase_a_driver_invocations": 0, "phase_a_worker_invocations": 0,
             "rng_proxy_delegate_invocations": 0, "committed_success": True},
        ]
        payload = b"".join(cbytes(line) for line in lines)
        os.lseek(descriptor, 0, os.SEEK_SET)
        write_all(descriptor, payload, writer)
        os.ftruncate(descriptor, len(payload))
        os.fsync(descriptor)
        observed = read_fd_all_exact(descriptor)
        if observed != payload or observed.splitlines(keepends=True) != [cbytes(line) for line in lines]:
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


def recursive_diff(left: object, right: object, path: tuple[object, ...] = ()) -> list[dict]:
    if type(left) is not type(right):
        return [{"path": list(path), "original": left, "normalized": right}]
    if isinstance(left, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows.append({"path": list(path + (key,)), "original": left.get(key), "normalized": right.get(key)})
            else:
                rows.extend(recursive_diff(left[key], right[key], path + (key,)))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": list(path), "original": left, "normalized": right}]
        rows = []
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(recursive_diff(a, b, path + (index,)))
        return rows
    return [] if left == right else [{"path": list(path), "original": left, "normalized": right}]


def validate_rng(value: dict, seed: int) -> None:
    stages = ("entry_external", "inside_before_delegate", "internal_after_delegate", "exit_restored")
    if (not isinstance(value, dict) or type(seed) is not int or type(value.get("exact_seed")) is not int
            or value.get("exact_seed") != seed or type(value.get("cuda_device_count")) is not int
            or value.get("cuda_device_count") != 1 or value.get("cuda_device_indices") != [0]
            or type(value.get("delegate_invocations_started")) is not int or value.get("delegate_invocations_started") != 1
            or type(value.get("delegate_invocations_completed")) is not int or value.get("delegate_invocations_completed") != 1):
        raise RuntimeError("RNG identity/count")
    for key in ("fork_rng_devices_exact_all_cuda_indices", "python_all_four_stages_equal",
                "numpy_all_four_stages_equal", "python_exit_restored", "numpy_exit_restored",
                "torch_cpu_exit_restored", "torch_cuda_exit_restored", "exception_finally_restoration_complete"):
        if value.get(key) is not True:
            raise RuntimeError(f"RNG flag {key}")
    if any(not isinstance(value.get(stage), dict) for stage in stages):
        raise RuntimeError("RNG stages")
    entry, inside, internal, exit_state = (value[stage] for stage in stages)
    if not (entry["python"] == inside["python"] == internal["python"] == exit_state["python"]):
        raise RuntimeError("python RNG four-stage")
    if not (entry["numpy"] == inside["numpy"] == internal["numpy"] == exit_state["numpy"]):
        raise RuntimeError("numpy RNG four-stage")
    if entry["torch"] != exit_state["torch"] or inside["torch"] != internal["torch"]:
        raise RuntimeError("torch RNG restoration")
    if value.get("four_stage_canonical_sha256") != csha({stage: value[stage] for stage in stages}):
        raise RuntimeError("RNG digest")


def validate_per_call() -> dict:
    result = {}
    warning = {"category": "UserWarning", "message": RAW_MEDIAN_WARNING}
    for name, role in (("process_a", "A"), ("process_b", "B")):
        events = (V524_QUALIFICATION / name / "call_events.ndjson").read_text().splitlines()
        markers = (V524_QUALIFICATION / name / "process.log").read_text().splitlines()
        if len(events) != 1000 or len(markers) != 2001:
            raise RuntimeError(f"event/marker counts {name}")
        if json.loads(markers[0]).get("event") != "worker_start":
            raise RuntimeError(f"worker marker {name}")
        for ordinal, line in enumerate(events):
            row = json.loads(line); unsigned = dict(row); digest = unsigned.pop("event_canonical_sha256", None)
            sample = ordinal if role == "A" else 999 - ordinal
            if (type(row.get("ordinal")) is not int or row.get("ordinal") != ordinal
                    or type(row.get("sample_id")) is not int or row.get("sample_id") != sample
                    or digest != csha(unsigned) or type(row.get("sample_call_started")) is not int
                    or row.get("sample_call_started") != 1 or type(row.get("sample_call_completed")) is not int
                    or row.get("sample_call_completed") != 1 or row.get("raw_warning") != warning
                    or row.get("warning_category") != "UserWarning" or row.get("warning_full_message") != RAW_MEDIAN_WARNING
                    or row.get("raw_warnings_unchanged") is not True):
                raise RuntimeError(f"event {name}:{ordinal}")
            validate_rng(row.get("rng_isolation"), row.get("seed"))
            if row.get("rng_isolation_sha256") != csha(row["rng_isolation"]) or row.get("rng_unchanged") is not True:
                raise RuntimeError(f"event RNG {name}:{ordinal}")
            started = json.loads(markers[1 + 2 * ordinal]); completed = json.loads(markers[2 + 2 * ordinal])
            if (started.get("event") != "rng_proxy_call_started" or started.get("ordinal") != ordinal
                    or started.get("sample_id") != sample or started.get("started") != 1 or started.get("completed") != 0
                    or completed.get("event") != "rng_proxy_call_completed" or completed.get("ordinal") != ordinal
                    or completed.get("sample_id") != sample or completed.get("started") != 1 or completed.get("completed") != 1
                    or completed.get("event_canonical_sha256") != digest):
                raise RuntimeError(f"marker {name}:{ordinal}")
        result[name] = {"role": role, "call_events_count": 1000, "started_markers": 1000,
                        "completed_markers": 1000, "total_markers": 2000,
                        "raw_median_user_warnings": 1000, "other_warnings": 0,
                        "python_numpy_four_stage_equal": 1000, "torch_cpu_cuda_exit_restored": 1000,
                        "delegate_started_completed": 1000, "cuda_device_indices": [0],
                        "process_log_marker_order": "strict_alternating_started_completed"}
    return result


def validate_success(module, contract: dict, returncode: int, cleanup: dict,
                     stdout_stream, stdout_identity: tuple[int, int], writer=os.write) -> dict:
    if not cleanup["reaped"] or not cleanup["group_empty"]:
        raise RuntimeError("reconciler cleanup")
    if not CANDIDATE_ROOT.is_dir() or os.path.lexists(CANDIDATE_PREP):
        raise RuntimeError(f"candidate absent/prep; child rc {returncode}")
    receipt_record = regular(CANDIDATE_RECEIPT)
    receipt = json.loads(CANDIDATE_RECEIPT.read_text())
    tree = exact_tree(CANDIDATE_ROOT)
    required = module.RECONCILER_CANDIDATE_CONTRACT
    if (tree["file_count"] != 1 or tree["inventory"] != [["reconciliation_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]
            or any(receipt.get(key) != value for key, value in required.items())
            or receipt.get("external_terminal_evidence_contract") != module.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT
            or receipt.get("readonly_reconciliation_completed") is not True
            or type(receipt.get("readonly_reconciler_invocations")) is not int or receipt.get("readonly_reconciler_invocations") != 1
            or any(type(receipt.get(key)) is not int or receipt.get(key) != 0 for key in (
                "phase_a_launcher_invocations", "phase_a_driver_invocations", "phase_a_worker_invocations",
                "rng_proxy_delegate_invocations", "training_invocations", "oof_invocations"))
            or any(receipt.get(key) is not False for key in ("retry_authorized", "training_authorized",
                "cache_reuse_authorized", "reward_read_authorized", "dev_hidden_final_outcome_read_authorized", "oof_authorized"))):
        raise RuntimeError("candidate nonterminal schema")
    original = receipt.get("original_worker_receipts"); normalized = receipt.get("normalized_worker_receipts")
    if not isinstance(original, dict) or not isinstance(normalized, dict) or set(original) != {"process_a", "process_b"}:
        raise RuntimeError("candidate normalization objects")
    rows = recursive_diff(original, normalized)
    expected_paths = [["process_a", "format"], ["process_b", "format"]]
    if (rows != receipt.get("recursive_diff") or [row["path"] for row in rows] != expected_paths
            or type(receipt.get("recursive_diff_count")) is not int or receipt.get("recursive_diff_count") != 2
            or receipt.get("original_worker_receipts_canonical_sha256") != csha(original)
            or receipt.get("normalized_worker_receipts_canonical_sha256") != csha(normalized)
            or receipt.get("recursive_diff_canonical_sha256") != csha(rows)
            or any(original[name].get("format") != OLD_FORMAT or normalized[name].get("format") != NORMALIZED_FORMAT
                   for name in ("process_a", "process_b"))):
        raise RuntimeError("candidate exact diff2")
    qualification_tree = exact_tree(V524_QUALIFICATION)
    if qualification_tree != contract.get("v524_qualification_tree") or receipt.get("v524_qualification_tree") != qualification_tree:
        raise RuntimeError("qualification exact20")
    per_call = validate_per_call()
    expected_per_call = receipt.get("per_call_validation")
    for name in ("process_a", "process_b"):
        if not isinstance(expected_per_call, dict) or any(expected_per_call.get(name, {}).get(key) != value for key, value in per_call[name].items()):
            raise RuntimeError("candidate per-call validation")
    if (receipt.get("input_snapshots_exactly_equal") is not True
            or receipt.get("input_pre_snapshot_sha256") != receipt.get("input_post_snapshot_sha256")
            or receipt.get("service_health_exactly_equal") is not True
            or receipt.get("service_health_pre") != receipt.get("service_health_post")
            or receipt.get("service_health_post") != service_health()
            or receipt.get("gpu_compute_pids") != [] or receipt.get("phase_a_pids") != []
            or gpu_compute_pids() or phase_a_or_reconciler_pids()):
        raise RuntimeError("candidate current/read-only gates")
    stdout_path = EVIDENCE_ROOT / "reconciler_stdout.log"
    diagnostic = {"returncode": returncode, "postrename_candidate_success_priority": returncode != 0,
                  "stderr": regular(EVIDENCE_ROOT / "reconciler_stderr.log")}
    stdout_lines, native_record = transport_stdout_exact2(
        stdout_stream, stdout_path, stdout_identity, receipt_record, tree, diagnostic, writer)
    return {"candidate_receipt": receipt_record, "candidate_tree": tree,
            "reconciler_native_stdout": native_record, "child_diagnostic": diagnostic,
            "qualification_tree": qualification_tree, "per_call_validation": per_call,
            "stdout_line_origin": "transport_helper",
            "transport_stdout_lines": stdout_lines}


def base_receipt(status: str, passed: bool, invocations: int, cleanup: dict, error: BaseException | None = None) -> dict:
    row = {
        "format": "strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-external-process-receipt-v1",
        "status": status, "passed": passed, "candidate_consumable": passed,
        "standalone_candidate_consumable": False, "external_terminal_required": True,
        "authority_materializer_invocations": 0,
        "transport_helper_invocations": 1,
        "readonly_reconciler_invocations": invocations,
        "phase_a_launcher_invocations": 0, "phase_a_driver_invocations": 0,
        "phase_a_worker_invocations": 0, "rng_proxy_delegate_invocations": 0,
        "retry_authorized": False, "cleanup": cleanup,
        "deployment_plan": deployment_plan(),
        "terminal_publication_contract": {
            "noreplace": True,
            "commit_semantics": "current_exact6_visibility_after_owned_temp_unlink",
            "crash_durability_claimed": False,
            "postcommit_parent_dir_fsync_best_effort": True,
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
    def visible():
        state["visible"] = True
        state["receipt"] = receipt
        state["terminal_record"] = regular(root / "process_receipt.json")
    def committed():
        state["committed"] = True
    atomic_json_noreplace(root / "process_receipt.json", receipt, visible, committed)
    observed = os.lstat(root)
    tree = exact_tree(root)
    expected_terminal = {
        "path": str(root / "process_receipt.json"),
        "sha256": hashlib.sha256(cbytes(receipt)).hexdigest(),
        "logical_bytes": len(cbytes(receipt)),
    }
    if ((observed.st_dev, observed.st_ino) != identity or tree["file_count"] != 6
            or [row[0] for row in tree["inventory"]] != EXACT6
            or os.path.lexists(root / ".process_receipt.json.noreplace-tmp")
            or state["terminal_record"] != expected_terminal
            or not records_current(receipt)):
        raise RuntimeError("terminal exact6")
    state["tree"] = tree
    phase_hook("terminal_visible")
    phase_hook("terminal_committed")
    return tree


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
             "terminal_record": None}
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
        stdout_descriptor = os.open(prep / "reconciler_stdout.log",
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
        prep_members["reconciler_stdout.log"] = owned_member(prep / "reconciler_stdout.log")
        stdout = os.fdopen(stdout_descriptor, "r+b", buffering=0)
        stderr_descriptor = os.open(prep / "reconciler_stderr.log",
                                    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        stderr_stat = os.fstat(stderr_descriptor)
        stderr_identity = (stderr_stat.st_dev, stderr_stat.st_ino)
        if not stat.S_ISREG(stderr_stat.st_mode) or stderr_stat.st_size != 0:
            os.close(stderr_descriptor)
            raise RuntimeError("stderr empty O_RDWR regular")
        os.fsync(stderr_descriptor)
        stderr = os.fdopen(stderr_descriptor, "r+b", buffering=0)
        prep_members["reconciler_stderr.log"] = owned_member(prep / "reconciler_stderr.log")
        promoted_argv_record = regular(prep / "argv.json")
        promoted_argv_record["path"] = str(root / "argv.json")
        write_exclusive(prep / "intent.json", cbytes({
            "format": "strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-external-transport-intent-v1",
            "status": "committed_before_exact_once_readonly_reconciler", "retry_authorized": False,
            "argv": promoted_argv_record,
        }))
        prep_members["intent.json"] = owned_member(prep / "intent.json")
        fsync_dir(prep)
        if [row[0] for row in exact_tree(prep)["inventory"]] != EXACT5:
            raise RuntimeError("prep exact5")
        assert_owned_fd(stdout, prep / "reconciler_stdout.log", stdout_identity)
        assert_owned_fd(stderr, prep / "reconciler_stderr.log", stderr_identity)
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
        assert_owned_fd(stdout, root / "reconciler_stdout.log", stdout_identity)
        assert_owned_fd(stderr, root / "reconciler_stderr.log", stderr_identity)
        if regular(root / "reconciler_stdout.log")["logical_bytes"] != 0:
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
        assert_owned_fd(stdout, root / "reconciler_stdout.log", stdout_identity)
        assert_owned_fd(stderr, root / "reconciler_stderr.log", stderr_identity)
        cleanup = terminate(process)
        extra = success_validator(returncode, cleanup, stdout, stdout_identity)
        close_fsync(stdout); stdout = None
        close_fsync(stderr); stderr = None
        phase_hook("candidate_exact1_before_flag")
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        state["authority_visible"] = True
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        phase_hook("after_authority_before_terminal")
        post_snapshot_diagnostic = None
        try:
            post_snapshot = argv_payload["pre_snapshot_builder"]()
            if post_snapshot != argv_payload["pre_snapshot"]:
                raise RuntimeError("immutable snapshot drift")
            snapshots_equal = True
        except BaseException as diagnostic_error:
            # The independently validated current exact1 candidate is the
            # success-priority boundary.  Later outer diagnostics cannot turn
            # it into a contradictory failed external terminal.
            post_snapshot = None
            snapshots_equal = None
            post_snapshot_diagnostic = {"error_type": type(diagnostic_error).__name__, "error": str(diagnostic_error),
                                        "success_priority_after_candidate_exact1": True}
        receipt = {
            **base_receipt("passed_external_terminal", True, 1, cleanup),
            "helper_returncode": 0, "reconciler_returncode": returncode,
            "transport_helper": helper_record, "transport_helper_copy": helper_copy,
            "transport_script": script_record, "argv": regular(root / "argv.json"),
            "intent": regular(root / "intent.json"), "stdout": regular(root / "reconciler_stdout.log"),
            "stderr": regular(root / "reconciler_stderr.log"),
            "pre_snapshot": argv_payload["pre_snapshot"], "post_snapshot": post_snapshot,
            "pre_post_snapshots_exactly_equal": snapshots_equal,
            "postcandidate_outer_diagnostic": post_snapshot_diagnostic,
            "stdout_descriptor_evidence": stdout_flags, **extra,
        }
        if failure_forensic_record is not None:
            receipt["failed_transport_forensic"] = failure_forensic_record
        # Builder callable is transport-local and never serialized.
        receipt["argv_payload_sha256"] = csha({key: value for key, value in argv_payload.items() if key != "pre_snapshot_builder"})
        tree = commit_terminal(root, receipt, baseline_mask, state, phase_hook)
        return {"passed": True, "process_receipt": receipt, "evidence_tree": tree}
    except BaseException as error:
        if state["committed"]:
            committed_receipt = state["receipt"]
            committed_tree = exact_tree(root)
            return {"passed": committed_receipt.get("passed") is True,
                    "process_receipt": committed_receipt, "evidence_tree": committed_tree,
                    "postpublish_error_type": type(error).__name__, "postpublish_error": str(error)}
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
                    ("reconciler_stdout.log", stdout_identity),
                    ("reconciler_stderr.log", stderr_identity)):
                metadata = os.lstat(root / log_name)
                if expected_identity is None or (metadata.st_dev, metadata.st_ino) != expected_identity:
                    raise RuntimeError("owned log identity drift; refusing terminal") from error
            if not state["committed"]:
                failure = {
                    **base_receipt("failed_no_retry", False, invocations, cleanup, error),
                    "transport_helper": helper_record,
                    "transport_helper_copy": regular(root / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"])),
                    "transport_script": script_record,
                    "stdout_descriptor_evidence": stdout_flags,
                    "argv": regular(root / "argv.json"), "intent": regular(root / "intent.json"),
                    "stdout": regular(root / "reconciler_stdout.log"), "stderr": regular(root / "reconciler_stderr.log"),
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
        "actual_stage_a_authority_contract_consumer": (
            contract_record["sha256"] == CONTRACT_SHA and materializer_record["sha256"] == MATERIALIZER_SHA),
        "final4_hardbound": all(value in source for value in FROZEN_FINAL4_SHA256),
        "final4_current_plan": (
            len(plan["stage_a_final4_current_exact"]) == 4
            and [row["role"] for row in plan["stage_a_final4_current_exact"]]
            == ["authority_design_contract", "authority_materializer", "readonly_reconciler", "failure_forensic"]),
        "sole_signal_owned_reconciler_popen": len(popens) == 1 and command()[1] == str(RECONCILER),
        "candidate_contract_exact": cbytes(contract["reconciler_candidate_contract"]) == cbytes(module.RECONCILER_CANDIDATE_CONTRACT),
        "external_terminal_contract_exact": cbytes(contract["external_terminal_evidence_contract"]) == cbytes(module.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT),
        "authority_schema_55_31_17": (
            len(contract["authority_receipt_contract"]["top_keys"]) == 55
            and len(module.CHECK_KEYS) == 31 and len(module.AUTHORITY_SOURCE_ORDER) == 17),
        "phase_a_not_in_child_argv": not any("launch_" in item or "generate_" in item for item in command()),
        "native_capture_to_transport_exact2_source": (
            "reconciler_native_capture" in source and "transport stdout exact2 readback" in source),
        "independent_per_call_2000_markers_4000": "def validate_per_call" in source and "len(events) != 1000" in source and "len(markers) != 2001" in source,
        "choice_a_external_evidence_exact6": (
            EXACT6 == sorted(EXACT5 + ["process_receipt.json"])
            and module.EXTERNAL_TERMINAL_EVIDENCE_CONTRACT["expected_file_count"] == 6),
    }
    with tempfile.TemporaryDirectory(prefix="v525-stage-b-helper-", dir="/dev/shm") as folder:
        temp = Path(folder)
        script = temp / "transport.sh"
        script.write_text("#!/bin/bash\n", encoding="utf-8")
        helper_record = regular(Path(__file__))
        script_record = regular(script)
        pre = {"stable": True}
        root = temp / "normal"
        result = orchestrate(
            root=root, prep=temp / "normal.execution-prep",
            child_argv=[str(RLPY), "-c", "pass"], helper_source=Path(__file__),
            helper_record=helper_record, script_record=script_record,
            argv_payload={"format": "synthetic-v525-stage-b", "pre_snapshot": pre,
                          "pre_snapshot_builder": lambda: pre}, timeout=30,
            success_validator=lambda rc, cleanup, stream, identity: {
                "synthetic_returncode": rc, "synthetic_cleanup": cleanup},
            retain_terminal_priority=True)
        checks["mapped_normal_exact6"] = (
            result["passed"] is True and exact_tree(root)["file_count"] == 6
            and result["process_receipt"]["readonly_reconciler_invocations"] == 1)
        fail_root = temp / "nonzero"
        failed = orchestrate(
            root=fail_root, prep=temp / "nonzero.execution-prep",
            child_argv=[str(RLPY), "-c", "raise SystemExit(9)"], helper_source=Path(__file__),
            helper_record=helper_record, script_record=script_record,
            argv_payload={"format": "synthetic-v525-stage-b", "pre_snapshot": pre,
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
            argv_payload={"format": "synthetic-v525-stage-b", "pre_snapshot": pre,
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
        target = primitive / "replacement.json"
        def replace_temp(path):
            path.unlink(); path.write_text("foreign", encoding="utf-8")
        try:
            atomic_json_noreplace(target, {"passed": True}, lambda: None, lambda: None,
                                  fixture_hook=replace_temp)
        except RuntimeError:
            checks["terminal_temp_replacement_rejected_preserved"] = (
                not target.exists() and target.with_name("." + target.name + ".noreplace-tmp").read_text() == "foreign")
        else:
            checks["terminal_temp_replacement_rejected_preserved"] = False
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
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"passed": True, "checks": checks, "checks_sha256": csha(checks),
            "materializer_invocations": 0, "readonly_reconciler_invocations": 0,
            "phase_a_invocations": 0}


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
    module, contract, contract_record, materializer_record = load_materializer()
    pre = immutable_snapshot(contract, helper_record, script_record)
    if pre["gpu_compute_pids"] or pre["phase_a_or_reconciler_pids"]:
        raise RuntimeError("PID/GPU prestate")
    if (not AUTHORITY_ROOT.is_dir() or os.path.lexists(AUTHORITY_PREP)
            or any(os.path.lexists(path) for path in (
                CANDIDATE_ROOT, CANDIDATE_PREP, EVIDENCE_ROOT, EVIDENCE_PREP))):
        raise RuntimeError("runtime prestate")
    if args.read_only_preflight:
        print(json.dumps({"passed": True, "read_only_preflight": True,
                          "contract": contract_record, "materializer": materializer_record,
                          "stable_snapshot_sha256": csha(pre)}, sort_keys=True))
        return 0
    argv = command()
    payload = {
        "format": "strict-track2-v525-v524-phase-a-worker-receipt-format-readonly-reconciliation-external-transport-argv-v1",
        "argv": argv, "argv_repr": repr(argv), "argv_utf8_hex": [item.encode().hex() for item in argv],
        "authority_contract": contract_record, "authority_materializer": materializer_record,
        "authority_receipt": regular(AUTHORITY_RECEIPT, (AUTHORITY_RECEIPT_SHA, AUTHORITY_RECEIPT_BYTES)),
        "readonly_reconciler": regular(RECONCILER, (RECONCILER_SHA, RECONCILER_BYTES)),
        "failure_forensic": regular(FAILURE_FORENSIC, (FAILURE_FORENSIC_SHA, FAILURE_FORENSIC_BYTES)),
        "runtime_prestate": stable_absences(contract),
        "transport_helper": helper_record,
        "pre_snapshot": pre, "pre_snapshot_builder": lambda: immutable_snapshot(contract, helper_record, script_record),
    }
    result = orchestrate(root=EVIDENCE_ROOT, prep=EVIDENCE_PREP, child_argv=argv,
                         helper_source=SELF_PATH, helper_record=helper_record, script_record=script_record,
                         argv_payload=payload, timeout=1800,
                         success_validator=lambda rc, clean, stream, identity: validate_success(
                             module, contract, rc, clean, stream, identity),
                         retain_terminal_priority=True)
    if not result["passed"]:
        raise RuntimeError(str(result["error"]))
    print(json.dumps({"passed": True, "evidence_tree": result["evidence_tree"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
