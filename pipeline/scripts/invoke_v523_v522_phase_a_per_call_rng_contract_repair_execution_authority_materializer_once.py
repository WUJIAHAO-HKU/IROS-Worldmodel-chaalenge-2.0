#!/usr/bin/env python3
"""Exact-once, signal-safe transport for the v523 exact21 repair authority materializer."""
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
SELF_PATH = SCRIPTS / "invoke_v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_materializer_once.py"
SCRIPT_PATH = Path("/root/v523_v522_phase_a_per_call_rng_contract_repair_authority_materialize_once.sh")
CONTRACT = SCRIPTS / "v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_contract.json"
CONTRACT_SHA = "e26f20b1a191f7e2711725786498bfaa0d88697d085410f4894eb3c5b2df2a8f"
CONTRACT_BYTES = 39453
MATERIALIZER = SCRIPTS / "materialize_v523_v522_phase_a_per_call_rng_contract_repair_execution_authority.py"
MATERIALIZER_SHA = "2b131db2f0fb95e24abc646930bcf88cae791ec5b9d696f287e55daf00f7c88a"
MATERIALIZER_BYTES = 33316
PHASE_A_LAUNCHER = SCRIPTS / "launch_v523_v522_phase_a_cache_qualification.py"
PHASE_A_LAUNCHER_SHA = "d207f80925b68830cbbf5a5e369b34b721a3831682f9d792cf3ed87d41eed7d3"
PHASE_A_LAUNCHER_BYTES = 81577
PHASE_A_DRIVER = SCRIPTS / "generate_v523_v522_phase_a_cache_qualification.py"
PHASE_A_DRIVER_SHA = "ee7a2a327cc7fc8f282eef9b97701860c310901cdf383d8bc923b7f276c4f3ab"
PHASE_A_DRIVER_BYTES = 47276
PHASE_A_WORKER = SCRIPTS / "generate_v523_v522_phase_a_cache_qualification_worker.py"
PHASE_A_WORKER_SHA = "89a552416c054fa2590d7f15b5fb829b8eb2cf8a02479e2115b825bf6e49d627"
PHASE_A_WORKER_BYTES = 38990
RNG_PROXY = SCRIPTS / "v520_v519_rng_isolated_v169_runtime_proxy.py"
RNG_PROXY_SHA = "2b8ab0e2aec4a8de5d4bf8cffbb3525bbcd69b9b03270012a4a48f69f430b51b"
RNG_PROXY_BYTES = 8926
INDEPENDENT_AUDITOR = SCRIPTS / "audit_v523_v522_phase_a_rng_isolated_qualification.py"
INDEPENDENT_AUDITOR_SHA = "f09a17f068f9f6f471be7bba10162cab406d18957d48586eed083e25d6f59ea2"
INDEPENDENT_AUDITOR_BYTES = 47453
PREREG = J / "v523_v522_phase_a_cache_qualification_prereg_seed1659_20260826/preregistration.json"
PREREG_SHA = "b813d87237a2a14e1b317da61497e5b563ebef2a6f0770ce25f023be66296ea2"
PREREG_BYTES = 57451
SPLIT_FORENSIC = SCRIPTS / "v523_v522_phase_a_per_call_rng_contract_failure_forensic.json"
SPLIT_FORENSIC_SHA = "94e06c8d9dcd31e3026e069dafe8b5bb6adbda85e51fd5e1e448f8dd4e6fab0d"
SPLIT_FORENSIC_BYTES = 5284
AUTHORITY_ROOT = J / "v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_seed1659_20260826"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")
EVIDENCE_ROOT = J / "v523_v522_phase_a_per_call_rng_contract_repair_execution_authority_materialization_evidence_seed1659_20260826"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
EXACT5 = ["argv.json", "intent.json", "materializer_stderr.log", "materializer_stdout.log", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])
AT_FDCWD = -100
RENAME_NOREPLACE = 1
FROZEN_FINAL9_SHA256 = (
    "e26f20b1a191f7e2711725786498bfaa0d88697d085410f4894eb3c5b2df2a8f",
    "2b131db2f0fb95e24abc646930bcf88cae791ec5b9d696f287e55daf00f7c88a",
    "d207f80925b68830cbbf5a5e369b34b721a3831682f9d792cf3ed87d41eed7d3",
    "ee7a2a327cc7fc8f282eef9b97701860c310901cdf383d8bc923b7f276c4f3ab",
    "89a552416c054fa2590d7f15b5fb829b8eb2cf8a02479e2115b825bf6e49d627",
    "2b8ab0e2aec4a8de5d4bf8cffbb3525bbcd69b9b03270012a4a48f69f430b51b",
    "f09a17f068f9f6f471be7bba10162cab406d18957d48586eed083e25d6f59ea2",
    "b813d87237a2a14e1b317da61497e5b563ebef2a6f0770ce25f023be66296ea2",
    "94e06c8d9dcd31e3026e069dafe8b5bb6adbda85e51fd5e1e448f8dd4e6fab0d",
)


def deployment_plan() -> dict:
    return {
        "fresh_noreplace_targets_exact8": [
            {"role": "authority_design_contract", "path": str(CONTRACT), "sha256": CONTRACT_SHA, "logical_bytes": CONTRACT_BYTES},
            {"role": "authority_materializer", "path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
            {"role": "fresh_phase_a_launcher", "path": str(PHASE_A_LAUNCHER), "sha256": PHASE_A_LAUNCHER_SHA, "logical_bytes": PHASE_A_LAUNCHER_BYTES},
            {"role": "fresh_phase_a_driver", "path": str(PHASE_A_DRIVER), "sha256": PHASE_A_DRIVER_SHA, "logical_bytes": PHASE_A_DRIVER_BYTES},
            {"role": "fresh_phase_a_worker", "path": str(PHASE_A_WORKER), "sha256": PHASE_A_WORKER_SHA, "logical_bytes": PHASE_A_WORKER_BYTES},
            {"role": "fresh_phase_a_independent_auditor", "path": str(INDEPENDENT_AUDITOR), "sha256": INDEPENDENT_AUDITOR_SHA, "logical_bytes": INDEPENDENT_AUDITOR_BYTES},
            {"role": "fresh_phase_a_preregistration", "path": str(PREREG), "sha256": PREREG_SHA, "logical_bytes": PREREG_BYTES},
            {"role": "v523_v522_phase_a_failure_forensic", "path": str(SPLIT_FORENSIC), "sha256": SPLIT_FORENSIC_SHA, "logical_bytes": SPLIT_FORENSIC_BYTES},
        ],
        "reused_current_only_exact1": [
            {"role": "fresh_rng_proxy", "path": str(RNG_PROXY), "sha256": RNG_PROXY_SHA, "logical_bytes": RNG_PROXY_BYTES},
        ],
        "preregistration_parent_directory": str(PREREG.parent),
        "preregistration_parent_directory_noreplace_required": True,
        "preregistration_file_and_parent_and_j_fsync_required": True,
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


def v523_phase_a_pids() -> list[dict]:
    markers = ("launch_v523_v522_phase_a_cache_qualification.py",
               "generate_v523_v522_phase_a_cache_qualification.py",
               "generate_v523_v522_phase_a_cache_qualification_worker.py")
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
    absences = {}
    for key, row in contract["current_absences_after_authority"].items():
        if not isinstance(row, dict) or set(row) != {"path"}:
            raise RuntimeError("contract absence schema")
        path = Path(row["path"])
        if path in {AUTHORITY_ROOT, AUTHORITY_PREP}:
            continue
        absences[key] = {"path": str(path), "absent": not os.path.lexists(path)}
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
    absences = stable_absences(contract)
    return {
        "contract": regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES)),
        "materializer": regular(MATERIALIZER, (MATERIALIZER_SHA, MATERIALIZER_BYTES)),
        "transport_helper": regular(SELF_PATH, (helper_record["sha256"], helper_record["logical_bytes"])),
        "transport_script": regular(SCRIPT_PATH, (script_record["sha256"], script_record["logical_bytes"])),
        "sources": sources, "trees": trees, "absences": absences,
        "service_health": service_health(), "gpu_compute_pids": gpu_compute_pids(),
        "v523_phase_a_pids": v523_phase_a_pids(), "deployment_plan": deployment_plan(),
    }


def validate_materializer_contract(module, contract: dict, materializer_record: dict) -> None:
    seed = contract.get("seed")
    active = {
        "authority_materializer": {"path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA, "logical_bytes": MATERIALIZER_BYTES},
        "fresh_phase_a_launcher": {"path": str(PHASE_A_LAUNCHER), "sha256": PHASE_A_LAUNCHER_SHA, "logical_bytes": PHASE_A_LAUNCHER_BYTES},
        "fresh_phase_a_driver": {"path": str(PHASE_A_DRIVER), "sha256": PHASE_A_DRIVER_SHA, "logical_bytes": PHASE_A_DRIVER_BYTES},
        "fresh_phase_a_worker": {"path": str(PHASE_A_WORKER), "sha256": PHASE_A_WORKER_SHA, "logical_bytes": PHASE_A_WORKER_BYTES},
        "fresh_rng_proxy": {"path": str(RNG_PROXY), "sha256": RNG_PROXY_SHA, "logical_bytes": RNG_PROXY_BYTES},
        "fresh_phase_a_independent_auditor": {"path": str(INDEPENDENT_AUDITOR), "sha256": INDEPENDENT_AUDITOR_SHA, "logical_bytes": INDEPENDENT_AUDITOR_BYTES},
        "fresh_phase_a_preregistration": {"path": str(PREREG), "sha256": PREREG_SHA, "logical_bytes": PREREG_BYTES},
        "v523_v522_phase_a_failure_forensic": {"path": str(SPLIT_FORENSIC), "sha256": SPLIT_FORENSIC_SHA, "logical_bytes": SPLIT_FORENSIC_BYTES},
    }
    if (module.CONTRACT_PATH != CONTRACT or module.AUTHORITY_ROOT != AUTHORITY_ROOT
            or set(contract) != module.CONTRACT_TOP_KEYS
            or type(seed) is not int or seed != 1659
            or contract.get("authority_materializer_source") != materializer_record
            or any(contract.get("source_closure", {}).get(role) != row for role, row in active.items())
            or len(contract.get("authority_receipt_contract", {}).get("top_keys", [])) != 71
            or len(module.CHECK_KEYS) != 33
            or len(module.CONTRACT_SOURCE_ORDER) != 26 or len(module.AUTHORITY_SOURCE_ORDER) != 27
            or contract.get("source_role_order") != module.CONTRACT_SOURCE_ORDER
            or contract.get("source_aliases") != {role: module.SOURCE_ALIASES[role] for role in module.CONTRACT_SOURCE_ORDER}):
        raise RuntimeError("materializer/contract schema")
    closure = contract["source_closure"]
    module.validate_prereg(closure)
    old = module.validate_old_state()
    old_keys = {
        "v522_authority_registration_tree": old["authority_tree"],
        "v522_authority_receipt": old["authority_receipt"],
        "v522_authority_semantics": old["authority_semantics"],
        "v522_attempt_tree": old["attempt_tree"],
        "v522_attempt_intent": old["attempt_intent"],
        "v522_attempt_stdout": old["attempt_stdout"],
        "v522_attempt_stderr": old["attempt_stderr"],
        "v522_attempt_terminal": old["attempt_terminal"],
        "v522_phase_a_failure_forensic": old["failure_forensic"],
        "v522_phase_a_execution_partition": old["execution_partition"],
        "v522_per_call_contract_failure_root_cause": old["root_cause"],
    }
    if (csha(contract.get("worker_environment_exact")) != csha(module.WORKER_ENV)
            or csha(contract.get("per_call_rng_evidence_contract")) != csha(module.PER_CALL_RNG_CONTRACT)
            or csha(contract.get("authorization")) != csha(module.AUTHORIZATION)
            or csha(contract.get("runtime_observation")) != csha(module.RUNTIME)
            or csha(contract.get("execution_boundary")) != csha(module.EXECUTION_BOUNDARY)
            or any(csha(contract.get(key)) != csha(value) for key, value in old_keys.items())):
        raise RuntimeError("v523 execution/failure-repair contract")


def load_materializer(contract_source: Path = CONTRACT, materializer_source: Path = MATERIALIZER,
                      prereg_source: Path | None = None):
    contract_record = regular(contract_source, (CONTRACT_SHA, CONTRACT_BYTES))
    materializer_record = regular(materializer_source, (MATERIALIZER_SHA, MATERIALIZER_BYTES))
    if contract_source != CONTRACT:
        contract_record = {**contract_record, "path": str(CONTRACT)}
    if materializer_source != MATERIALIZER:
        materializer_record = {**materializer_record, "path": str(MATERIALIZER)}
    contract = json.loads(contract_source.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("v523_materializer", materializer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if prereg_source is not None:
        regular(prereg_source, (PREREG_SHA, PREREG_BYTES))
        module.PREREG_PATH = prereg_source
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
        native_record = {
            "line_origin": "materializer_native_empty", "logical_bytes": len(native),
            "sha256": hashlib.sha256(native).hexdigest(), "eof_exact": True,
        }
        if native != b"" or native_record["sha256"] != hashlib.sha256(b"").hexdigest():
            raise RuntimeError("materializer native stdout must be empty")
        lines = [
            {"line_origin": "transport_helper", "materializer_native_empty": True,
             "authority_receipt": authority_record, "authority_registration_tree": authority_tree},
            {"line_origin": "transport_helper", "passed": True,
             "transport_helper_invocations": 1, "authority_materializer_invocations": 1,
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
    if (os.path.lexists(AUTHORITY_PREP)
            or sorted(receipt) != receipt_schema["top_keys"] or len(receipt) != 71
            or receipt.get("format") != module.OUTPUT_FORMAT
            or receipt.get("status") != module.OUTPUT_STATUS or receipt.get("passed") is not True
            or len(receipt.get("checks", {})) != 33
            or set(receipt.get("checks", {})) != set(module.CHECK_KEYS)
            or not all(type(value) is bool and value is True for value in receipt["checks"].values())
            or receipt.get("check_key_set_sha256") != csha(module.CHECK_KEYS)
            or receipt.get("checks_sha256") != csha(receipt["checks"])
            or len(receipt.get("source_closure", {})) != 27
            or receipt.get("source_role_order") != module.AUTHORITY_SOURCE_ORDER
            or receipt.get("source_aliases") != module.SOURCE_ALIASES
            or receipt.get("input_snapshots_exactly_equal") is not True
            or receipt.get("input_pre_snapshot") != receipt.get("input_post_snapshot")
            or csha(receipt.get("authorization")) != csha(module.AUTHORIZATION)
            or csha(receipt.get("runtime_observation")) != csha(module.RUNTIME)
            or csha(receipt.get("execution_boundary")) != csha(module.EXECUTION_BOUNDARY)
            or receipt.get("source_closure_sha256") != csha(receipt.get("source_closure"))
            or receipt.get("authority_design_contract") != regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES))
            or tree["inventory"] != [["authority_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]):
        raise RuntimeError("authority receipt")
    for row in receipt["source_closure"].values():
        if regular(Path(row["path"]), (row["sha256"], row["logical_bytes"])) != row:
            raise RuntimeError("authority source current")
    split_keys = (
        "v522_authority_registration_tree", "v522_authority_receipt", "v522_authority_semantics",
        "v522_attempt_tree", "v522_attempt_intent", "v522_attempt_stdout", "v522_attempt_stderr",
        "v522_attempt_terminal", "v522_phase_a_failure_forensic", "v522_phase_a_execution_partition",
        "v522_per_call_contract_failure_root_cause",
    )
    if any(receipt.get(key) != contract.get(key) for key in split_keys):
        raise RuntimeError("authority failure-repair anchors")
    for row in receipt.get("required_absences", {}).values():
        if row.get("absent") is not True or os.path.lexists(row.get("path", "")):
            raise RuntimeError("downstream absence")
    stdout_lines, native_record = transport_stdout_exact2(
        stdout_stream, stdout_path, stdout_identity, receipt_record, tree, writer)
    return {"authority_receipt": receipt_record, "authority_registration_tree": tree,
            "materializer_native_stdout": native_record,
            "stdout_line_origin": "transport_helper",
            "transport_stdout_lines": stdout_lines}


def base_receipt(status: str, passed: bool, invocations: int, cleanup: dict, error: BaseException | None = None) -> dict:
    row = {
        "format": "strict-track2-v523-v522-phase-a-per-call-rng-contract-repair-execution-authority-materializer-process-receipt-v1",
        "status": status, "passed": passed, "authority_materializer_invocations": invocations,
        "transport_helper_invocations": 1,
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
            "format": "strict-track2-v523-v522-phase-a-per-call-rng-contract-repair-execution-authority-materializer-intent-v1",
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
            **base_receipt("passed_exact_once_authority_materialized_no_phase_a_execution", True, 1, cleanup),
            "helper_returncode": 0, "materializer_returncode": returncode,
            "transport_helper": helper_record, "transport_helper_copy": helper_copy,
            "transport_script": script_record, "argv": regular(root / "argv.json"),
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
                       prereg_source: Path | None = None) -> dict:
    checks = {}
    live_module, live_contract, live_contract_record, live_materializer_record = load_materializer(
        contract_source, materializer_source, prereg_source)
    checks["actual_main_load_materializer"] = (
        live_contract_record == {"path": str(CONTRACT), "sha256": CONTRACT_SHA, "logical_bytes": CONTRACT_BYTES}
        and live_materializer_record == {"path": str(MATERIALIZER), "sha256": MATERIALIZER_SHA,
                                         "logical_bytes": MATERIALIZER_BYTES}
        and live_contract["seed"] == 1659)
    source = Path(__file__).read_text(encoding="utf-8")
    syntax = ast.parse(source)
    popens = [node for node in ast.walk(syntax) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"]
    checks["sole_signal_owned_materializer_popen"] = (
        len(popens) == 1 and "pthread_sigmask(signal.SIG_BLOCK" in source
        and "start_new_session=True" in source and "child_argv" in source)
    checks["final9_hardbound"] = all(value in source for value in FROZEN_FINAL9_SHA256)
    plan = deployment_plan()
    checks["fresh8_proxy_current_only_and_prereg_parent_plan"] = (
        len(plan["fresh_noreplace_targets_exact8"]) == 8
        and len(plan["reused_current_only_exact1"]) == 1
        and plan["reused_current_only_exact1"][0] == {
            "role": "fresh_rng_proxy", "path": str(RNG_PROXY),
            "sha256": RNG_PROXY_SHA, "logical_bytes": RNG_PROXY_BYTES}
        and plan["preregistration_parent_directory"] == str(PREREG.parent)
        and plan["preregistration_parent_directory_noreplace_required"] is True
        and plan["preregistration_file_and_parent_and_j_fsync_required"] is True)
    checks["choice_a_visibility_commit_contract"] = all(value in source for value in (
        "current_exact6_visibility_after_owned_temp_unlink", "crash_durability_claimed",
        "postcommit_parent_dir_fsync_best_effort", "no_rollback_relink_or_second_publish"))
    checks["services_pid_gpu_in_stable_snapshot"] = all(value in source for value in (
        '"service_health": service_health()', '"gpu_compute_pids": gpu_compute_pids()',
        '"v523_phase_a_pids": v523_phase_a_pids()'))
    stable_rows = stable_absences(live_contract)
    checks["stable_snapshot_excludes_dynamic_authority_root_and_prep"] = (
        all(row["path"] not in {str(AUTHORITY_ROOT), str(AUTHORITY_PREP)}
            for row in stable_rows.values())
        and all(row["absent"] is True for row in stable_rows.values()))
    old_state = live_module.validate_old_state()
    checks["old_v522_authority_and_failed_attempt_current"] = (
        exact_tree(Path(live_contract["v522_authority_registration_tree"]["root"]))
            == live_contract["v522_authority_registration_tree"] == old_state["authority_tree"]
        and exact_tree(Path(live_contract["v522_attempt_tree"]["root"]))
            == live_contract["v522_attempt_tree"] == old_state["attempt_tree"])
    def rejects(contract_value, module_value=live_module, materializer_value=live_materializer_record):
        try:
            validate_materializer_contract(module_value, contract_value, materializer_value)
            return False
        except RuntimeError:
            return True
    tamper_checks = {}
    for bad_seed in (1658, 1660, "1659", True, None):
        tampered = copy.deepcopy(live_contract); tampered["seed"] = bad_seed
        tamper_checks[f"seed_{bad_seed!r}"] = rejects(tampered)
    missing = copy.deepcopy(live_contract); missing.pop("seed")
    extra = copy.deepcopy(live_contract); extra["unexpected"] = True
    tamper_checks["missing_top"] = rejects(missing)
    tamper_checks["extra_top"] = rejects(extra)
    bad_source = copy.deepcopy(live_contract)
    bad_source["source_closure"]["fresh_phase_a_launcher"]["sha256"] = "0" * 64
    tamper_checks["launcher_source"] = rejects(bad_source)
    bad_materializer = copy.deepcopy(live_contract)
    bad_materializer["authority_materializer_source"]["logical_bytes"] += 1
    tamper_checks["materializer_alias"] = rejects(bad_materializer)
    bad_worker = copy.deepcopy(live_contract)
    bad_worker["source_closure"]["fresh_phase_a_worker"]["sha256"] = "0" * 64
    tamper_checks["worker_source"] = rejects(bad_worker)
    bad_bool_int = copy.deepcopy(live_contract)
    bad_bool_int["authorization"]["launcher_invocations_authorized"] = True
    tamper_checks["authorization_bool_int"] = rejects(bad_bool_int)
    bad_partition_bool = copy.deepcopy(live_contract)
    bad_partition_bool["v522_phase_a_execution_partition"]["launcher_invocations"] = True
    tamper_checks["partition_bool_int"] = rejects(bad_partition_bool)
    module_values = {name: getattr(live_module, name) for name in (
        "CONTRACT_PATH", "AUTHORITY_ROOT", "CONTRACT_TOP_KEYS", "CHECK_KEYS",
        "CONTRACT_SOURCE_ORDER", "AUTHORITY_SOURCE_ORDER", "SOURCE_ALIASES", "WORKER_ENV",
        "PER_CALL_RNG_CONTRACT", "AUTHORIZATION", "RUNTIME", "EXECUTION_BOUNDARY",
        "validate_prereg", "validate_old_state")}
    bad_module = types.SimpleNamespace(**module_values)
    bad_module.CONTRACT_SOURCE_ORDER = list(live_module.CONTRACT_SOURCE_ORDER[:-1])
    tamper_checks["module_source_order"] = rejects(copy.deepcopy(live_contract), bad_module)
    checks["actual_load_materializer_tamper_suite"] = all(tamper_checks.values()) and len(tamper_checks) == 13
    for name, passed in tamper_checks.items():
        checks[f"contract_tamper_rejected_{name}"] = passed
    with tempfile.TemporaryDirectory(prefix="v523-helper-fixture-") as folder:
        base = Path(folder)
        helper = base / "helper.py"; helper.write_bytes(Path(__file__).read_bytes()); helper_record = regular(helper)
        script = base / "run.sh"; script.write_text("#!/bin/bash\n"); script_record = regular(script)
        def run(name, code, timeout=3, hook=None, missing=False, retain_terminal_priority=False):
            parent = base / name; parent.mkdir(); root = parent / "evidence"; prep = parent / "evidence.execution-prep"
            child = parent / "child.py"; child.write_text(code)
            argv = [str(parent / "missing")] if missing else [sys.executable, str(child)]
            pre = {"fixture": name}
            payload = {"format": "fixture", "argv": argv, "pre_snapshot": pre, "pre_snapshot_builder": lambda: pre}
            def fixture_success(rc, clean, stdout_stream, stdout_owned_identity):
                if rc != 0 or not clean["reaped"] or not clean["group_empty"]:
                    raise RuntimeError(f"rc {rc}")
                fixture_authority = parent / "authority"
                fixture_authority.mkdir()
                (fixture_authority / "authority_receipt.json").write_bytes(b"")
                fixture_record = regular(fixture_authority / "authority_receipt.json")
                fixture_tree = exact_tree(fixture_authority)
                lines, native = transport_stdout_exact2(
                    stdout_stream, root / "materializer_stdout.log", stdout_owned_identity,
                    fixture_record, fixture_tree)
                return {"fixture": True, "transport_stdout_lines": lines,
                        "materializer_native_stdout": native, "stdout_line_origin": "transport_helper"}
            result = orchestrate(root=root, prep=prep, child_argv=argv, helper_source=helper,
                                 helper_record=helper_record, script_record=script_record,
                                 argv_payload=payload, timeout=timeout,
                                 success_validator=fixture_success,
                                 phase_hook=hook, retain_terminal_priority=retain_terminal_priority)
            return result, root, prep
        result, root, prep = run("success", "pass\n")
        success_result = result
        checks["success_exact6_all_current"] = result["passed"] and result["evidence_tree"]["file_count"] == 6 and records_current(result["process_receipt"]) and not os.path.lexists(prep)
        checks["stdout_child_descriptor_flags_odrdw"] = result["process_receipt"]["stdout_descriptor_evidence"]["access_mode"] == "O_RDWR" and result["process_receipt"]["stdout_descriptor_evidence"]["held_from_before_popen_through_child_wait"] is True
        stdout_path = root / "materializer_stdout.log"
        descriptor = os.open(stdout_path, os.O_RDWR)
        try:
            observed = os.pread(descriptor, stdout_path.stat().st_size + 1, 0)
            checks["stdout_odrdw_exact2_eof"] = len(observed.splitlines()) == 2 and os.pread(descriptor, 1, len(observed)) == b""
        finally:
            os.close(descriptor)
        if "transport_stdout_lines" not in result["process_receipt"]:
            raise RuntimeError({"success_fixture_result": result})
        checks["stdout_exact2_transport_origin"] = (
            len(result["process_receipt"]["transport_stdout_lines"]) == 2
            and all(row["line_origin"] == "transport_helper"
                    for row in result["process_receipt"]["transport_stdout_lines"])
            and result["process_receipt"]["materializer_native_stdout"] == {
                "line_origin": "materializer_native_empty", "logical_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(), "eof_exact": True}
        )
        # Native child output is forbidden even when it is canonical JSON.
        result, _, _ = run("native_nonempty", "print('{}')\n")
        checks["native_nonempty_rejected"] = not result["passed"] and "native stdout" in result["process_receipt"]["error"]
        def stdout_write_fixture(name, writer, should_pass):
            path = base / f"{name}.log"
            path.write_bytes(b"")
            record = {"path": "/fixture/authority_receipt.json", "sha256": "0" * 64,
                      "logical_bytes": 0}
            tree = {"root": "/fixture/authority", "inventory": [], "file_count": 1,
                    "logical_file_bytes": 0, "sha256sum_lines_digest_sha256": "1" * 64,
                    "canonical_json_triples_digest_sha256": "2" * 64}
            try:
                descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                stream = os.fdopen(descriptor, "r+b", buffering=0)
                metadata = os.fstat(descriptor)
                try:
                    lines, native = transport_stdout_exact2(
                        stream, path, (metadata.st_dev, metadata.st_ino), record, tree, writer)
                finally:
                    stream.close()
                return should_pass and len(lines) == 2 and native["logical_bytes"] == 0
            except RuntimeError:
                return not should_pass
        checks["stdout_partial_write_all"] = stdout_write_fixture(
            "partial", lambda fd, data: os.write(fd, data[:max(1, min(7, len(data)))]), True)
        checks["stdout_zero_write_rejected"] = stdout_write_fixture("zero", lambda _fd, _data: 0, False)
        checks["stdout_invalid_long_write_rejected"] = stdout_write_fixture(
            "invalid_long", lambda _fd, data: len(data) + 1, False)
        def postauthority_signal(phase):
            if phase == "authority_exact1_before_flag":
                os.kill(os.getpid(), signal.SIGTERM)
        result, signal_root, _ = run("postauthority_signal", "pass\n", hook=postauthority_signal)
        checks["postauthority_signal_success_priority"] = (
            result["passed"] and exact_tree(signal_root)["file_count"] == 6
            and records_current(result["process_receipt"])
        )
        checks["helper_copy_final_path"] = success_result["process_receipt"]["transport_helper_copy"]["path"] == str(root / "transport_helper.py")
        checks["intent_argv_final_path_current"] = records_current(json.loads((root / "intent.json").read_text()))
        result, _, _ = run("nonzero", "raise SystemExit(7)\n")
        checks["nonzero_failure_exact6"] = not result["passed"] and result["evidence_tree"]["file_count"] == 6 and records_current(result["process_receipt"])
        result, _, _ = run("popen", "pass\n", missing=True)
        checks["popen_failure_exact6"] = not result["passed"] and result["process_receipt"]["authority_materializer_invocations"] == 0 and records_current(result["process_receipt"])
        result, _, _ = run("timeout", "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n", timeout=.2)
        checks["timeout_kill_reap"] = not result["passed"] and result["process_receipt"]["cleanup"]["kill_sent"] and result["process_receipt"]["cleanup"]["group_empty"]
        def pending(phase):
            if phase == "after_promote_before_unmask": os.kill(os.getpid(), signal.SIGTERM)
        result, _, _ = run("signal", "print('never')\n", hook=pending)
        checks["pending_signal_failure_exact6"] = not result["passed"] and records_current(result["process_receipt"])
        def preintent(phase):
            if phase == "prep_owned": raise RuntimeError("preintent")
        try:
            run("preintent", "pass\n", hook=preintent)
            checks["preintent_cleanup"] = False
        except RuntimeError:
            checks["preintent_cleanup"] = not os.path.lexists(base / "preintent/evidence") and not os.path.lexists(base / "preintent/evidence.execution-prep")
        foreign_root_parent = base / "foreign_promote"
        def foreign_promote(phase):
            if phase == "before_promote":
                (foreign_root_parent / "evidence").mkdir()
        try:
            run("foreign_promote", "pass\n", hook=foreign_promote)
            checks["evidence_root_noreplace_foreign_preserved"] = False
        except FileExistsError:
            foreign_root = foreign_root_parent / "evidence"
            checks["evidence_root_noreplace_foreign_preserved"] = (
                foreign_root.is_dir() and not list(foreign_root.iterdir())
                and not os.path.lexists(foreign_root_parent / "evidence.execution-prep"))
        log_replacement_checks = []
        for log_name in ("materializer_stdout.log", "materializer_stderr.log"):
            fixture_name = "replace_" + log_name.split("_")[1].split(".")[0]
            replacement_root = base / fixture_name / "evidence"
            def replace_log(phase, selected=log_name, target_root=replacement_root):
                if phase == "after_promote_before_unmask":
                    target = target_root / selected
                    target.unlink()
                    target.write_bytes(b"foreign-log-replacement")
            try:
                run(fixture_name, "pass\n", hook=replace_log)
                log_replacement_checks.append(False)
            except RuntimeError:
                target = replacement_root / log_name
                log_replacement_checks.append(
                    target.read_bytes() == b"foreign-log-replacement"
                    and not os.path.lexists(replacement_root / "process_receipt.json"))
        checks["stdout_stderr_replacements_refuse_write_or_terminal"] = all(log_replacement_checks)
        replacement_checks = []
        for index, member_name in enumerate(EXACT5):
            replacement_prep = base / f"replacement-{index}"
            replacement_prep.mkdir()
            prep_metadata = os.lstat(replacement_prep)
            member_path = replacement_prep / member_name
            member_path.write_bytes(b"owned")
            frozen = {member_name: owned_member(member_path)}
            member_path.unlink()
            member_path.write_bytes(b"foreign replacement")
            replacement_record = regular(member_path)
            try:
                cleanup_owned_prep(replacement_prep, (prep_metadata.st_dev, prep_metadata.st_ino), frozen)
                replacement_checks.append(False)
            except RuntimeError:
                replacement_checks.append(
                    replacement_prep.is_dir() and regular(member_path) == replacement_record)
        checks["prep_member_replacements_refuse_cleanup"] = all(replacement_checks)
        terminal_temp_root = base / "terminal_temp_replacement"
        terminal_temp_root.mkdir()
        terminal_target = terminal_temp_root / "process_receipt.json"
        replacement_payload = b"foreign-terminal-temp"
        def replace_terminal_temp(path):
            path.unlink()
            path.write_bytes(replacement_payload)
        try:
            atomic_json_noreplace(terminal_target, {"owned": True}, lambda: None, lambda: None, replace_terminal_temp)
            checks["terminal_temp_replacement_refuses_cleanup"] = False
        except RuntimeError:
            temp = terminal_temp_root / ".process_receipt.json.noreplace-tmp"
            checks["terminal_temp_replacement_refuses_cleanup"] = (
                not os.path.lexists(terminal_target) and temp.read_bytes() == replacement_payload)
        # A foreign or already-committed terminal is immutable: a second commit
        # must fail before publication and preserve the first terminal byte-for-byte.
        foreign_sha = regular(root / "process_receipt.json")["sha256"]
        try:
            commit_terminal(root, {"foreign": True}, set(),
                            {"visible": False, "committed": False, "tree": None,
                             "receipt": None, "terminal_record": None},
                            lambda _phase: None)
            checks["foreign_terminal_noreplace"] = False
        except RuntimeError:
            checks["foreign_terminal_noreplace"] = regular(root / "process_receipt.json")["sha256"] == foreign_sha
        def terminal_fault_case(name, fault_global):
            original_fault = globals()[fault_global]
            original_publish = globals()["atomic_json_noreplace"]
            publish_calls = {"value": 0}
            def counted_publish(*args, **kwargs):
                publish_calls["value"] += 1
                return original_publish(*args, **kwargs)
            def injected_fault(_path):
                raise OSError(f"injected {fault_global}")
            globals()["atomic_json_noreplace"] = counted_publish
            globals()[fault_global] = injected_fault
            fault_root = base / name / "evidence"
            raised = False
            result = None
            try:
                result, _, _ = run(name, "pass\n")
            except OSError:
                raised = True
            finally:
                globals()[fault_global] = original_fault
                globals()["atomic_json_noreplace"] = original_publish
            terminal = fault_root / "process_receipt.json"
            terminal_sha = regular(terminal)["sha256"] if terminal.is_file() and not terminal.is_symlink() else None
            return {
                "raised_nonzero": raised,
                "publish_attempts": publish_calls["value"],
                "terminal_sha256": terminal_sha,
                "terminal_sha256_readback": regular(terminal)["sha256"] if terminal_sha is not None else None,
                "temp_exists": os.path.lexists(fault_root / ".process_receipt.json.noreplace-tmp"),
                "file_count": exact_tree(fault_root)["file_count"],
                "passed": result is not None and result["passed"],
            }
        unlink_fault = terminal_fault_case("terminal_unlink_fault", "unlink_terminal_temp")
        checks["unlink_failure_visible_not_committed_no_second_publish"] = (
            unlink_fault["raised_nonzero"] and unlink_fault["publish_attempts"] == 1
            and unlink_fault["terminal_sha256"] == unlink_fault["terminal_sha256_readback"]
            and unlink_fault["temp_exists"] and unlink_fault["file_count"] == 7
        )
        fsync_fault = terminal_fault_case("terminal_fsync_fault", "fsync_terminal_parent")
        checks["fsync_failure_postcommit_success_priority_no_second_publish"] = (
            not fsync_fault["raised_nonzero"] and fsync_fault["passed"] and fsync_fault["publish_attempts"] == 1
            and fsync_fault["terminal_sha256"] == fsync_fault["terminal_sha256_readback"]
            and not fsync_fault["temp_exists"] and fsync_fault["file_count"] == 6
        )
        exception_visible_sha = {"value": None}
        exception_root = base / "postpublish_exception/evidence"
        def postpublish_exception(phase):
            if phase == "terminal_visible":
                exception_visible_sha["value"] = regular(exception_root / "process_receipt.json")["sha256"]
                raise RuntimeError("postpublish fixture")
        result, post_root, _ = run(
            "postpublish_exception",
            "pass\n",
            hook=postpublish_exception,
        )
        checks["postpublish_exception_preserves_terminal"] = (
            result["passed"] and result.get("postpublish_error_type") == "RuntimeError"
            and exception_visible_sha["value"] is not None
            and regular(post_root / "process_receipt.json")["sha256"] == exception_visible_sha["value"]
            and result["evidence_tree"]["file_count"] == 6
        )
        signal_visible_sha = {"value": None}
        signal_fixture_root = base / "postpublish_signal/evidence"
        def postpublish_signal(phase):
            if phase == "terminal_visible":
                signal_visible_sha["value"] = regular(signal_fixture_root / "process_receipt.json")["sha256"]
                os.kill(os.getpid(), signal.SIGTERM)
        result, signal_root, _ = run(
            "postpublish_signal",
            "pass\n",
            hook=postpublish_signal,
        )
        checks["postpublish_signal_preserves_terminal"] = (
            result["passed"] and signal_visible_sha["value"] is not None
            and regular(signal_root / "process_receipt.json")["sha256"] == signal_visible_sha["value"]
            and result["evidence_tree"]["file_count"] == 6
        )
        # Production main keeps committed-success priority through its final
        # print/exit window.  Exercise that exact retained state, then restore it
        # locally so the remaining self-test process is well behaved.
        exit_old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            result, exit_root, _ = run(
                "after_orchestrate_before_exit",
                "pass\n",
                retain_terminal_priority=True,
            )
            exit_sha = regular(exit_root / "process_receipt.json")["sha256"]
            os.kill(os.getpid(), signal.SIGTERM)
            checks["after_orchestrate_before_exit_success_priority"] = (
                result["passed"]
                and signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
                and regular(exit_root / "process_receipt.json")["sha256"] == exit_sha
            )
        finally:
            for sig, handler in exit_old.items():
                signal.signal(sig, handler)
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {"passed": True, "checks": checks, "checks_sha256": csha(checks)}


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
    if pre["gpu_compute_pids"] or pre["v523_phase_a_pids"]:
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
        "format": "strict-track2-v523-v522-phase-a-per-call-rng-contract-repair-execution-authority-materializer-argv-v1",
        "argv": argv, "argv_repr": repr(argv), "argv_utf8_hex": [item.encode().hex() for item in argv],
        "authority_contract": contract_record, "authority_materializer": materializer_record,
        "transport_helper": helper_record,
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
    print(json.dumps({"passed": True, "evidence_tree": result["evidence_tree"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
