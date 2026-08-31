#!/usr/bin/env python3
"""Exact-once, signal-safe transport for the v499 read-only receipt adapter."""
from __future__ import annotations

import argparse
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
SELF_PATH = SCRIPTS / "invoke_v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_once.py"
SCRIPT_PATH = Path("/root/v499_diagnostic_adapter_once.sh")
ADAPTER_SOURCE = SCRIPTS / "adapt_v499_v498_diagnostic_process_receipt_transport_helper_copy_path.py"
ADAPTER_SOURCE_SHA = "5456dfd102d08b2825bb2ba2d6ea44e01c5c5d443823799115c9f3ff105ed35c"
ADAPTER_SOURCE_BYTES = 28726
ADAPTER_ROOT = J / "v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_seed1641_20260825"
ADAPTER_PREP = ADAPTER_ROOT.with_name(ADAPTER_ROOT.name + ".registration-prep")
EVIDENCE_ROOT = J / "v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_execution_evidence_seed1641_20260825"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
EXACT5 = ["adapter_stderr.log", "adapter_stdout.log", "argv.json", "intent.json", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])
ADAPTER_TOP_COUNT = 32
ADAPTER_TOP_KEY_SET_SHA = "656b80abeed36a815550fc48edd51093a6754b5c593895331cc22214a7388bee"


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
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
            raise RuntimeError(f"tree member: {path}")
        inventory.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError("terminal temp exists")
    write_exclusive(temporary, cbytes(value))
    os.replace(temporary, path)
    fsync_dir(path.parent)


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


def immutable_snapshot(adapter_source_record: dict, helper_record: dict, script_record: dict) -> dict:
    return {
        "adapter_source": regular(ADAPTER_SOURCE, (ADAPTER_SOURCE_SHA, ADAPTER_SOURCE_BYTES)),
        "transport_helper": regular(SELF_PATH, (helper_record["sha256"], helper_record["logical_bytes"])),
        "transport_script": regular(SCRIPT_PATH, (script_record["sha256"], script_record["logical_bytes"])),
        "adapter_source_matches_argv": regular(ADAPTER_SOURCE) == adapter_source_record,
    }


def load_adapter():
    record = regular(ADAPTER_SOURCE, (ADAPTER_SOURCE_SHA, ADAPTER_SOURCE_BYTES))
    spec = importlib.util.spec_from_file_location("v499_adapter", ADAPTER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SELF_PATH != ADAPTER_SOURCE or len(module.RECEIPT_TOP_KEYS) != ADAPTER_TOP_COUNT or csha(sorted(module.RECEIPT_TOP_KEYS)) != ADAPTER_TOP_KEY_SET_SHA:
        raise RuntimeError("adapter source schema")
    return module, record


def command() -> list[str]:
    return [str(RLPY), str(ADAPTER_SOURCE), "--adapter-source", str(ADAPTER_SOURCE),
            "--adapter-source-sha", ADAPTER_SOURCE_SHA, "--output-root", str(ADAPTER_ROOT)]


def validate_success(module, returncode: int, cleanup: dict, adapter_source_record: dict) -> dict:
    if returncode != 0 or not cleanup["reaped"] or not cleanup["group_empty"]:
        raise RuntimeError(f"adapter child rc {returncode}")
    receipt_record = regular(ADAPTER_ROOT / "adapter_receipt.json")
    receipt = json.loads(Path(receipt_record["path"]).read_text(encoding="utf-8"))
    if (set(receipt) != module.RECEIPT_TOP_KEYS or receipt.get("format") != "strict-track2-v499-v498-diagnostic-process-receipt-helper-copy-path-adapter-v1"
            or receipt.get("status") != "passed_readonly_exact_one_leaf_process_receipt_normalization_no_authority"
            or receipt.get("passed") is not True or receipt.get("adapter_source") != adapter_source_record
            or receipt.get("input_snapshots_exactly_equal") is not True
            or not (receipt.get("input_pre_snapshot_1") == receipt.get("input_pre_snapshot_2") == receipt.get("input_post_snapshot_1") == receipt.get("input_post_snapshot_2"))
            or receipt.get("authorization") != module.AUTHORIZATION or any(receipt["authorization"].values())
            or receipt.get("runtime_observation") != module.RUNTIME
            or receipt.get("canonical_json_leaf_diff_count") != 1
            or receipt.get("normalization_exactly_one_leaf") is not True):
        raise RuntimeError("adapter receipt")
    tree = exact_tree(ADAPTER_ROOT)
    if tree["inventory"] != [["adapter_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]:
        raise RuntimeError("adapter exact1")
    return {"adapter_receipt": receipt_record, "adapter_registration_tree": tree}


def base_receipt(status: str, passed: bool, invocations: int, cleanup: dict, error: BaseException | None = None) -> dict:
    row = {
        "format": "strict-track2-v499-diagnostic-process-receipt-adapter-transport-v1",
        "status": status, "passed": passed, "adapter_invocations": invocations,
        "authority_materializer_invocations": 0, "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0, "reconciler_r2_invocations": 0,
        "retry_authorized": False, "cleanup": cleanup,
    }
    if error is not None:
        row.update({"error_type": type(error).__name__, "error": str(error)})
    return row


def commit_terminal(root: Path, receipt: dict, baseline_mask, state: dict, phase_hook) -> dict:
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        phase_hook("before_terminal")
        atomic_json(root / "process_receipt.json", receipt)
        tree = exact_tree(root)
        if tree["file_count"] != 6 or [row[0] for row in tree["inventory"]] != EXACT6:
            raise RuntimeError("terminal exact6")
        if not records_current(receipt):
            raise RuntimeError("terminal path records not current")
        state["committed"] = True
        state["tree"] = tree
        phase_hook("terminal_committed")
        return tree
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)


def orchestrate(*, root: Path, prep: Path, child_argv: list[str], helper_source: Path,
                helper_record: dict, script_record: dict, argv_payload: dict,
                timeout: float, success_validator, phase_hook=None) -> dict:
    phase_hook = phase_hook or (lambda _phase: None)
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("evidence prestate")
    prep_identity = None
    prep_owned = False
    promoted = False
    process = None
    stdout = None
    stderr = None
    invocations = 0
    state = {"committed": False, "tree": None}
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
        write_exclusive(prep / "adapter_stdout.log", b"")
        write_exclusive(prep / "adapter_stderr.log", b"")
        promoted_argv_record = regular(prep / "argv.json")
        promoted_argv_record["path"] = str(root / "argv.json")
        write_exclusive(prep / "intent.json", cbytes({
            "format": "strict-track2-v499-diagnostic-process-receipt-adapter-intent-v1",
            "status": "committed_before_exact_once_readonly_adapter", "retry_authorized": False,
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
        stdout = (root / "adapter_stdout.log").open("ab", buffering=0)
        stderr = (root / "adapter_stderr.log").open("ab", buffering=0)
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
            **base_receipt("passed_exact_once_readonly_adapter_no_authority", True, 1, cleanup),
            "helper_returncode": 0, "adapter_returncode": returncode,
            "transport_helper": helper_record, "transport_helper_copy": helper_copy,
            "transport_script": script_record, "argv": regular(root / "argv.json"),
            "intent": regular(root / "intent.json"), "stdout": regular(root / "adapter_stdout.log"),
            "stderr": regular(root / "adapter_stderr.log"),
            "pre_snapshot": argv_payload["pre_snapshot"], "post_snapshot": post_snapshot,
            "pre_post_snapshots_exactly_equal": True, **extra,
        }
        # Builder callable is transport-local and never serialized.
        receipt["argv_payload_sha256"] = csha({key: value for key, value in argv_payload.items() if key != "pre_snapshot_builder"})
        tree = commit_terminal(root, receipt, baseline_mask, state, phase_hook)
        return {"passed": True, "process_receipt": receipt, "evidence_tree": tree}
    except BaseException as error:
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
                    "argv": regular(root / "argv.json"), "intent": regular(root / "intent.json"),
                    "stdout": regular(root / "adapter_stdout.log"), "stderr": regular(root / "adapter_stderr.log"),
                }
                tree = commit_terminal(root, failure, baseline_mask, state, phase_hook)
                return {"passed": False, "process_receipt": failure, "evidence_tree": tree, "error": error}
        if prep_owned:
            cleanup_owned_prep(prep, prep_identity)
        raise
    finally:
        try: signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        except BaseException: pass
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def production_fixture() -> dict:
    checks = {}
    with tempfile.TemporaryDirectory(prefix="v499-helper-fixture-") as folder:
        base = Path(folder)
        helper = base / "helper.py"; helper.write_bytes(Path(__file__).read_bytes()); helper_record = regular(helper)
        script = base / "run.sh"; script.write_text("#!/bin/bash\n"); script_record = regular(script)
        def run(name, code, timeout=3, hook=None, missing=False):
            parent = base / name; parent.mkdir(); root = parent / "evidence"; prep = parent / "evidence.execution-prep"
            child = parent / "child.py"; child.write_text(code)
            argv = [str(parent / "missing")] if missing else [sys.executable, str(child)]
            pre = {"fixture": name}
            payload = {"format": "fixture", "argv": argv, "pre_snapshot": pre, "pre_snapshot_builder": lambda: pre}
            result = orchestrate(root=root, prep=prep, child_argv=argv, helper_source=helper,
                                 helper_record=helper_record, script_record=script_record,
                                 argv_payload=payload, timeout=timeout,
                                 success_validator=lambda rc, clean: ({"fixture": True} if rc == 0 and clean["reaped"] and clean["group_empty"] else (_ for _ in ()).throw(RuntimeError(f"rc {rc}"))),
                                 phase_hook=hook)
            return result, root, prep
        result, root, prep = run("success", "print('ok')\n")
        checks["success_exact6_all_current"] = result["passed"] and result["evidence_tree"]["file_count"] == 6 and records_current(result["process_receipt"]) and not os.path.lexists(prep)
        checks["helper_copy_final_path"] = result["process_receipt"]["transport_helper_copy"]["path"] == str(root / "transport_helper.py")
        checks["intent_argv_final_path_current"] = records_current(json.loads((root / "intent.json").read_text()))
        result, _, _ = run("nonzero", "raise SystemExit(7)\n")
        checks["nonzero_failure_exact6"] = not result["passed"] and result["evidence_tree"]["file_count"] == 6 and records_current(result["process_receipt"])
        result, _, _ = run("popen", "pass\n", missing=True)
        checks["popen_failure_exact6"] = not result["passed"] and result["process_receipt"]["adapter_invocations"] == 0 and records_current(result["process_receipt"])
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
    module, adapter_record = load_adapter()
    pre = immutable_snapshot(adapter_record, helper_record, script_record)
    if any(os.path.lexists(path) for path in (ADAPTER_ROOT, ADAPTER_PREP, EVIDENCE_ROOT, EVIDENCE_PREP)):
        raise RuntimeError("runtime prestate")
    argv = command()
    payload = {
        "format": "strict-track2-v499-diagnostic-process-receipt-adapter-argv-v1",
        "argv": argv, "argv_repr": repr(argv), "argv_utf8_hex": [item.encode().hex() for item in argv],
        "adapter_source": adapter_record, "transport_helper": helper_record,
        "pre_snapshot": pre, "pre_snapshot_builder": lambda: immutable_snapshot(adapter_record, helper_record, script_record),
    }
    result = orchestrate(root=EVIDENCE_ROOT, prep=EVIDENCE_PREP, child_argv=argv,
                         helper_source=SELF_PATH, helper_record=helper_record, script_record=script_record,
                         argv_payload=payload, timeout=300,
                         success_validator=lambda rc, clean: validate_success(module, rc, clean, adapter_record))
    if not result["passed"]:
        raise RuntimeError(str(result["error"]))
    print(json.dumps({"passed": True, "evidence_tree": result["evidence_tree"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
