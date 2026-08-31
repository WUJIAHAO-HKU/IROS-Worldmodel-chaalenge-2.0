#!/usr/bin/env python3
"""Exact-once, signal-safe transport for the v502 inner-binding-path-repair authority materializer."""
from __future__ import annotations

import argparse
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
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS = ROOT / "pipeline/scripts"
RLPY = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
SELF_PATH = SCRIPTS / "invoke_v502_v501_inner_binding_path_repair_execution_authority_materializer_once.py"
SCRIPT_PATH = Path("/root/v502_inner_binding_path_repair_authority_materialize_once.sh")
CONTRACT = SCRIPTS / "v502_v501_inner_binding_path_repair_execution_authority_contract.json"
CONTRACT_SHA = "f4686ddf218cf809e35d0410e8eabd7eb9400a4ed11adf48519284cf7a161d9f"
CONTRACT_BYTES = 163229
MATERIALIZER = SCRIPTS / "materialize_v502_v501_inner_binding_path_repair_execution_authority.py"
MATERIALIZER_SHA = "2d339b62a1187d87c22fcd4e1c585b2a7d0c42fe14049ab0534031b365557ff0"
MATERIALIZER_BYTES = 146883
INNER_WRAPPER = SCRIPTS / "launch_v502_v501_inner_binding_path_repaired_reconciliation_inner.py"
INNER_WRAPPER_SHA = "db8d3f4d5c98e484a7fb6f26fad4a62cb056e0c42f801de31892eff267702075"
INNER_WRAPPER_BYTES = 154171
OUTER_WRAPPER = SCRIPTS / "launch_v502_v501_inner_binding_path_repaired_reconciliation_outer.py"
OUTER_WRAPPER_SHA = "d13655df4430a3975d56b97e80cd59c15f0f67573dc060db1ab2e189b01c1a1a"
OUTER_WRAPPER_BYTES = 112571
V502_FORENSIC = SCRIPTS / "v502_v501_outer_preintent_materializer_path_mismatch_failure_forensic.json"
V502_FORENSIC_SHA = "ce8d13769f56d4e0f501fbb8e1691c221ee2dc554eb26cabcb0861b0a5ab1f21"
V502_FORENSIC_BYTES = 10908
AUTHORITY_ROOT = J / "v502_v501_inner_binding_path_repair_execution_authority_seed1644_20260825"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")
EVIDENCE_ROOT = J / "v502_v501_inner_binding_path_repair_execution_authority_materialization_evidence_seed1644_20260825"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
EXACT5 = ["argv.json", "intent.json", "materializer_stderr.log", "materializer_stdout.log", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])


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


def atomic_json_noreplace(path: Path, value: object, visible_hook) -> None:
    temporary = path.with_name("." + path.name + ".noreplace-tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise FileExistsError(path)
    write_exclusive(temporary, cbytes(value))
    linked = False
    try:
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        visible_hook()
        unlink_terminal_temp(temporary)
        fsync_terminal_parent(path.parent)
    except BaseException:
        if not linked and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
            fsync_dir(path.parent)
        raise


def close_fsync(stream) -> None:
    if stream is None or stream.closed:
        return
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()


def cleanup_owned_prep(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None or not path.is_dir() or path.is_symlink():
        return
    observed = os.lstat(path)
    if (observed.st_dev, observed.st_ino) != identity:
        return
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError("foreign prep member")
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
    absences = {}
    for key, row in contract["current_absences_after_authority"].items():
        if not isinstance(row, dict) or set(row) != {"path"}:
            raise RuntimeError("contract absence schema")
        path = Path(row["path"])
        absences[key] = {"path": str(path), "absent": not os.path.lexists(path)}
    if not all(row["absent"] for row in absences.values()):
        raise RuntimeError("contract current absence")
    return {
        "contract": regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES)),
        "materializer": regular(MATERIALIZER, (MATERIALIZER_SHA, MATERIALIZER_BYTES)),
        "transport_helper": regular(SELF_PATH, (helper_record["sha256"], helper_record["logical_bytes"])),
        "transport_script": regular(SCRIPT_PATH, (script_record["sha256"], script_record["logical_bytes"])),
        "sources": sources, "trees": trees, "absences": absences,
    }


def load_materializer():
    contract_record = regular(CONTRACT, (CONTRACT_SHA, CONTRACT_BYTES))
    materializer_record = regular(MATERIALIZER, (MATERIALIZER_SHA, MATERIALIZER_BYTES))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("v502_materializer", MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.CONTRACT_PATH != CONTRACT or module.AUTHORITY_ROOT != AUTHORITY_ROOT
            or set(contract) != module.CONTRACT_TOP_KEYS or contract.get("seed") != 1644
            or contract.get("authority_materializer_source") != materializer_record
            or contract.get("source_closure", {}).get("corrected_inner_wrapper") != {
                "path": str(INNER_WRAPPER), "sha256": INNER_WRAPPER_SHA, "logical_bytes": INNER_WRAPPER_BYTES}
            or contract.get("source_closure", {}).get("outer_execution_wrapper") != {
                "path": str(OUTER_WRAPPER), "sha256": OUTER_WRAPPER_SHA, "logical_bytes": OUTER_WRAPPER_BYTES}
            or contract.get("source_closure", {}).get("v502_preexecution_forensic_source") != {
                "path": str(V502_FORENSIC), "sha256": V502_FORENSIC_SHA, "logical_bytes": V502_FORENSIC_BYTES}
            or len(module.AUTH_TOP_KEYS) != 122 or len(module.AUTH_CHECK_KEYS) != 108
            or len(module.SOURCE_ROLE_ORDER) != 46
            or contract.get("source_role_order") != module.SOURCE_ROLE_ORDER
            or contract.get("source_aliases") != module.SOURCE_ALIASES):
        raise RuntimeError("materializer/contract schema")
    return module, contract, contract_record, materializer_record


def command() -> list[str]:
    return [str(RLPY), str(MATERIALIZER), "--contract", str(CONTRACT),
            "--contract-sha", CONTRACT_SHA, "--materializer-source", str(MATERIALIZER),
            "--materializer-sha", MATERIALIZER_SHA, "--authority-root", str(AUTHORITY_ROOT)]


def parse_stdout_exact2(path: Path, checkpoint_contract: dict) -> tuple[dict, dict]:
    descriptor = os.open(path, os.O_RDWR)
    try:
        size = os.fstat(descriptor).st_size
        payload = os.pread(descriptor, size + 1, 0)
        if len(payload) != size or os.pread(descriptor, 1, size) != b"":
            raise RuntimeError("materializer stdout EOF")
    finally:
        os.close(descriptor)
    lines = payload.splitlines(keepends=True)
    if len(lines) != 2 or any(not line.endswith(b"\n") for line in lines):
        raise RuntimeError("materializer stdout exact2")
    values = [json.loads(line) for line in lines]
    if any(cbytes(value) != line for value, line in zip(values, lines)):
        raise RuntimeError("materializer stdout canonical lines")
    checkpoint_envelope, summary = values
    if (set(checkpoint_envelope) != {"checkpoint", "checkpoint_sha256"}
            or checkpoint_envelope["checkpoint_sha256"] != csha(checkpoint_envelope["checkpoint"])
            or checkpoint_envelope["checkpoint"].get("format") != checkpoint_contract["format"]
            or set(checkpoint_envelope["checkpoint"]) != set(checkpoint_contract["top_keys"])
            or checkpoint_envelope["checkpoint"].get("passed") is not True
            or checkpoint_envelope["checkpoint"].get("false_predicate_names") != []):
        raise RuntimeError("materializer checkpoint line")
    return checkpoint_envelope, summary


def validate_success(module, contract: dict, returncode: int, cleanup: dict) -> dict:
    if returncode != 0 or not cleanup["reaped"] or not cleanup["group_empty"]:
        raise RuntimeError(f"materializer child rc {returncode}")
    stdout_path = EVIDENCE_ROOT / "materializer_stdout.log"
    checkpoint_envelope, summary = parse_stdout_exact2(stdout_path, contract["materializer_checkpoint_contract"])
    receipt_record = regular(AUTHORITY_ROOT / "authority_receipt.json")
    receipt = json.loads(Path(receipt_record["path"]).read_text(encoding="utf-8"))
    tree = exact_tree(AUTHORITY_ROOT)
    expected_summary_keys = {"path", "sha256", "logical_bytes", "authority_registration_tree",
                             "committed_success", "deferred_signal", "outer_execution_wrapper_invocations",
                             "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"}
    if (set(receipt) != module.AUTH_TOP_KEYS or receipt.get("format") != module.OUTPUT_FORMAT
            or receipt.get("status") != module.OUTPUT_STATUS or receipt.get("passed") is not True
            or receipt.get("checks") != {key: True for key in module.AUTH_CHECK_KEYS}
            or receipt.get("input_snapshots_exactly_equal") is not True
            or receipt.get("authorization") != module.AUTHORIZATION
            or receipt.get("runtime_observation") != module.RUNTIME
            or receipt.get("materializer_pre_root_checkpoint") != checkpoint_envelope["checkpoint"]
            or receipt.get("materializer_pre_root_checkpoint_sha256") != checkpoint_envelope["checkpoint_sha256"]
            or tree["inventory"] != [["authority_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]
            or set(summary) != expected_summary_keys or summary["path"] != receipt_record["path"]
            or summary["sha256"] != receipt_record["sha256"] or summary["logical_bytes"] != receipt_record["logical_bytes"]
            or summary["authority_registration_tree"] != tree or summary["committed_success"] is not True
            or any(summary[key] != 0 for key in ("outer_execution_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))):
        raise RuntimeError("authority receipt/stdout summary")
    return {"authority_receipt": receipt_record, "authority_registration_tree": tree,
            "checkpoint": checkpoint_envelope, "success_summary": summary}


def base_receipt(status: str, passed: bool, invocations: int, cleanup: dict, error: BaseException | None = None) -> dict:
    row = {
        "format": "strict-track2-v502-inner-binding-path-repair-authority-materializer-process-receipt-v1",
        "status": status, "passed": passed, "authority_materializer_invocations": invocations,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0, "reconciler_r2_invocations": 0,
        "retry_authorized": False, "cleanup": cleanup,
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
        # The no-replace link is now observable, but it is not yet a durable
        # exact6 commit: the temp link still has to be removed and the parent
        # directory fsynced.  This state prohibits every second publish attempt.
        state["visible"] = True
        state["receipt"] = receipt
        state["terminal_record"] = regular(root / "process_receipt.json")
    atomic_json_noreplace(root / "process_receipt.json", receipt, visible)
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
    state["committed"] = True
    phase_hook("terminal_visible")
    phase_hook("terminal_committed")
    return tree


def orchestrate(*, root: Path, prep: Path, child_argv: list[str], helper_source: Path,
                 helper_record: dict, script_record: dict, argv_payload: dict,
                 timeout: float, success_validator, phase_hook=None,
                 retain_terminal_priority: bool = False) -> dict:
    phase_hook = phase_hook or (lambda _phase: None)
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("evidence prestate")
    prep_identity = None
    prep_owned = False
    promoted = False
    process = None
    stdout = None
    stderr = None
    stdout_flags = None
    invocations = 0
    state = {"visible": False, "committed": False, "tree": None, "receipt": None,
             "terminal_record": None}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, _frame):
        if state["committed"]:
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
        serialized_argv = {key: value for key, value in argv_payload.items() if key != "pre_snapshot_builder"}
        write_exclusive(prep / "argv.json", cbytes({**serialized_argv, "transport_script": script_record}))
        stdout_descriptor = os.open(prep / "materializer_stdout.log", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        stdout_stat = os.fstat(stdout_descriptor)
        observed_flags = fcntl.fcntl(stdout_descriptor, fcntl.F_GETFL)
        if (not stat.S_ISREG(stdout_stat.st_mode) or stdout_stat.st_size != 0
                or observed_flags & os.O_ACCMODE != os.O_RDWR):
            os.close(stdout_descriptor)
            raise RuntimeError("stdout empty O_RDWR regular")
        stdout_flags = {"open_flags": observed_flags, "access_mode": "O_RDWR",
                        "empty_regular_before_child": True, "held_from_before_popen_through_child_wait": True}
        os.fsync(stdout_descriptor)
        stdout = os.fdopen(stdout_descriptor, "r+b", buffering=0)
        write_exclusive(prep / "materializer_stderr.log", b"")
        promoted_argv_record = regular(prep / "argv.json")
        promoted_argv_record["path"] = str(root / "argv.json")
        write_exclusive(prep / "intent.json", cbytes({
            "format": "strict-track2-v502-inner-binding-path-repair-authority-materializer-intent-v1",
            "status": "committed_before_exact_once_authority_materializer", "retry_authorized": False,
            "argv": promoted_argv_record,
        }))
        fsync_dir(prep)
        if [row[0] for row in exact_tree(prep)["inventory"]] != EXACT5:
            raise RuntimeError("prep exact5")
        phase_hook("before_promote")
        os.replace(prep, root)
        fsync_dir(root.parent)
        prep_owned = False
        promoted = True
        # Rebuild after promote: this exact record must name the final persistent path.
        helper_copy = regular(root / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"]))
        phase_hook("after_promote_before_unmask")
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        if regular(root / "materializer_stdout.log")["logical_bytes"] != 0:
            raise RuntimeError("stdout prechild empty")
        stderr = (root / "materializer_stderr.log").open("ab", buffering=0)
        spawn_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            phase_hook("before_popen")
            process = subprocess.Popen(child_argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, close_fds=True)
            invocations = 1
            phase_hook("popen_owned")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)
        returncode = process.wait(timeout=timeout)
        close_fsync(stdout); stdout = None
        close_fsync(stderr); stderr = None
        cleanup = terminate(process)
        extra = success_validator(returncode, cleanup)
        post_snapshot = argv_payload["pre_snapshot_builder"]()
        if post_snapshot != argv_payload["pre_snapshot"]:
            raise RuntimeError("immutable snapshot drift")
        receipt = {
            **base_receipt("passed_exact_once_authority_materialized_no_outer_or_inner_execution", True, 1, cleanup),
            "helper_returncode": 0, "materializer_returncode": returncode,
            "transport_helper": helper_record, "transport_helper_copy": helper_copy,
            "transport_script": script_record, "argv": regular(root / "argv.json"),
            "intent": regular(root / "intent.json"), "stdout": regular(root / "materializer_stdout.log"),
            "stderr": regular(root / "materializer_stderr.log"),
            "pre_snapshot": argv_payload["pre_snapshot"], "post_snapshot": post_snapshot,
            "pre_post_snapshots_exactly_equal": True, "stdout_descriptor_evidence": stdout_flags, **extra,
        }
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
            cleanup = terminate(process)
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
                tree = commit_terminal(root, failure, baseline_mask, state, phase_hook)
                return {"passed": False, "process_receipt": failure, "evidence_tree": tree, "error": error}
        if prep_owned:
            cleanup_owned_prep(prep, prep_identity)
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


def production_fixture() -> dict:
    checks = {}
    with tempfile.TemporaryDirectory(prefix="v502-helper-fixture-") as folder:
        base = Path(folder)
        helper = base / "helper.py"; helper.write_bytes(Path(__file__).read_bytes()); helper_record = regular(helper)
        script = base / "run.sh"; script.write_text("#!/bin/bash\n"); script_record = regular(script)
        def run(name, code, timeout=3, hook=None, missing=False, retain_terminal_priority=False):
            parent = base / name; parent.mkdir(); root = parent / "evidence"; prep = parent / "evidence.execution-prep"
            child = parent / "child.py"; child.write_text(code)
            argv = [str(parent / "missing")] if missing else [sys.executable, str(child)]
            pre = {"fixture": name}
            payload = {"format": "fixture", "argv": argv, "pre_snapshot": pre, "pre_snapshot_builder": lambda: pre}
            result = orchestrate(root=root, prep=prep, child_argv=argv, helper_source=helper,
                                 helper_record=helper_record, script_record=script_record,
                                 argv_payload=payload, timeout=timeout,
                                 success_validator=lambda rc, clean: ({"fixture": True} if rc == 0 and clean["reaped"] and clean["group_empty"] else (_ for _ in ()).throw(RuntimeError(f"rc {rc}"))),
                                 phase_hook=hook, retain_terminal_priority=retain_terminal_priority)
            return result, root, prep
        result, root, prep = run("success", "import json\nprint(json.dumps({'checkpoint':{},'checkpoint_sha256':'x'},sort_keys=True,separators=(',',':')))\nprint(json.dumps({'committed_success':True},sort_keys=True,separators=(',',':')))\n")
        checks["success_exact6_all_current"] = result["passed"] and result["evidence_tree"]["file_count"] == 6 and records_current(result["process_receipt"]) and not os.path.lexists(prep)
        checks["stdout_child_descriptor_flags_odrdw"] = result["process_receipt"]["stdout_descriptor_evidence"]["access_mode"] == "O_RDWR" and result["process_receipt"]["stdout_descriptor_evidence"]["held_from_before_popen_through_child_wait"] is True
        stdout_path = root / "materializer_stdout.log"
        descriptor = os.open(stdout_path, os.O_RDWR)
        try:
            observed = os.pread(descriptor, stdout_path.stat().st_size + 1, 0)
            checks["stdout_odrdw_exact2_eof"] = len(observed.splitlines()) == 2 and os.pread(descriptor, 1, len(observed)) == b""
        finally:
            os.close(descriptor)
        parser_path = base / "stdout-parser.log"
        checkpoint = {"format": "fixture", "passed": True, "false_predicate_names": []}
        envelope = {"checkpoint": checkpoint, "checkpoint_sha256": csha(checkpoint)}
        summary = {"committed_success": True}
        parser_path.write_bytes(cbytes(envelope) + cbytes(summary))
        parsed_checkpoint, parsed_summary = parse_stdout_exact2(
            parser_path, {"format": "fixture", "top_keys": sorted(checkpoint)})
        checks["stdout_exact2_json_checkpoint_success"] = parsed_checkpoint == envelope and parsed_summary == summary
        parser_path.write_bytes(cbytes(envelope) + cbytes(summary) + cbytes({"extra": True}))
        try:
            parse_stdout_exact2(parser_path, {"format": "fixture", "top_keys": sorted(checkpoint)})
            checks["stdout_extra_line_rejected"] = False
        except RuntimeError:
            checks["stdout_extra_line_rejected"] = True
        checks["helper_copy_final_path"] = result["process_receipt"]["transport_helper_copy"]["path"] == str(root / "transport_helper.py")
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
            try:
                run(name, "print('ok')\n")
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
            }
        unlink_fault = terminal_fault_case("terminal_unlink_fault", "unlink_terminal_temp")
        checks["unlink_failure_visible_not_committed_no_second_publish"] = (
            unlink_fault["raised_nonzero"] and unlink_fault["publish_attempts"] == 1
            and unlink_fault["terminal_sha256"] == unlink_fault["terminal_sha256_readback"]
            and unlink_fault["temp_exists"] and unlink_fault["file_count"] == 7
        )
        fsync_fault = terminal_fault_case("terminal_fsync_fault", "fsync_terminal_parent")
        checks["fsync_failure_visible_not_committed_no_second_publish"] = (
            fsync_fault["raised_nonzero"] and fsync_fault["publish_attempts"] == 1
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
            "print('ok')\n",
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
            "print('ok')\n",
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
                "print('ok')\n",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        print(json.dumps(production_fixture(), sort_keys=True)); return 0
    helper_record = regular(SELF_PATH, (args.self_sha, args.self_bytes))
    script_record = regular(SCRIPT_PATH, (args.script_sha, args.script_bytes))
    module, contract, contract_record, materializer_record = load_materializer()
    pre = immutable_snapshot(contract, helper_record, script_record)
    if any(os.path.lexists(path) for path in (AUTHORITY_ROOT, AUTHORITY_PREP, EVIDENCE_ROOT, EVIDENCE_PREP)):
        raise RuntimeError("runtime prestate")
    argv = command()
    payload = {
        "format": "strict-track2-v502-inner-binding-path-repair-authority-materializer-argv-v1",
        "argv": argv, "argv_repr": repr(argv), "argv_utf8_hex": [item.encode().hex() for item in argv],
        "authority_contract": contract_record, "authority_materializer": materializer_record,
        "transport_helper": helper_record,
        "pre_snapshot": pre, "pre_snapshot_builder": lambda: immutable_snapshot(contract, helper_record, script_record),
    }
    result = orchestrate(root=EVIDENCE_ROOT, prep=EVIDENCE_PREP, child_argv=argv,
                         helper_source=SELF_PATH, helper_record=helper_record, script_record=script_record,
                         argv_payload=payload, timeout=300,
                         success_validator=lambda rc, clean: validate_success(module, contract, rc, clean),
                         retain_terminal_priority=True)
    if not result["passed"]:
        raise RuntimeError(str(result["error"]))
    print(json.dumps({"passed": True, "evidence_tree": result["evidence_tree"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
