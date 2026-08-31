#!/usr/bin/env python3
"""Signal-safe, exact-once transport for the v497 pre-authority diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SELF_PATH = SCRIPTS / "invoke_v497_v496_v495_failure_tree_pre_authority_diagnostic_once.py"
TRANSPORT_SCRIPT_PATH = Path("/root/v497_pre_authority_diagnostic_once.sh")
DIAGNOSTIC_SOURCE = SCRIPTS / "diagnose_v497_v496_v495_failure_tree_pre_authority.py"
DIAGNOSTIC_SOURCE_SHA = "6ad8f54b53fc2178330f27e9bc772de8fc7357a27752f17a309b835f8f796ecb"
DIAGNOSTIC_SOURCE_BYTES = 26978
DIAGNOSTIC_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1639_20260825"
DIAGNOSTIC_PREP = DIAGNOSTIC_ROOT.with_name(DIAGNOSTIC_ROOT.name + ".registration-prep")
EVIDENCE_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1639_20260825"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".execution-prep")
V496_FAILURE_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_materialization_evidence_seed1638_20260825"
V496_TREE = {
    "root": str(V496_FAILURE_ROOT),
    "inventory": [
        ["argv.json", "327cb32d890f937f7b0fec53902cf721af8df9a6195de9d82c01fbad7afb5f8b", 18856],
        ["intent.json", "c5bb3b6759f53a1de617f42cfbc343e6caf2dcb94916caba8cf3047bc025d06a", 557],
        ["materializer_stderr.log", "1ce5cb2055a08736e0a4927af4817f08bbd1c1e421e132574deff4bc70a471fa", 531],
        ["materializer_stdout.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["process_receipt.json", "5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d", 411],
        ["transport_helper.py", "df90feed74fa40eb0dba758cb285be9766fed0ca6ff8a79285a468ea40a23305", 14530],
    ],
    "file_count": 6, "logical_file_bytes": 34885,
    "sha256sum_lines_digest_sha256": "ab63f37f6275193a97db39cc71d37b32f4f653f87e78c375ddb460558b4e1f9e",
    "canonical_json_triples_digest_sha256": "c78038212aee67743f08dc936a539ad5d99a6ff931b87f12b5b294dfa4a0dac0",
}
V496_PROCESS_SHA = "5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d"
RECEIPT_TOP_KEYS = {
    "format", "status", "passed", "diagnostic_source", "v496_authority_sources",
    "v496_failure_tree", "v496_failure_process_receipt", "v496_failure_transport_helper",
    "v496_failure_no_retry", "root_cause_boundary", "predicate_names",
    "predicate_name_key_set_sha256", "predicates", "predicates_sha256",
    "all_predicates_passed", "snapshot_1", "snapshot_2", "snapshots_exactly_equal",
    "required_absences", "runtime_observation", "authorization",
}
EXACT5 = ["argv.json", "diagnostic_stderr.log", "diagnostic_stdout.log", "intent.json", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])


class ControlledSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"controlled signal {signum}")
        self.signum = signum


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def regular(path: Path | str, expected: tuple[str, int] | None = None) -> dict:
    path = Path(path)
    if path != path.resolve() or path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(f"regular: {path}")
    row = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if expected is not None and (row["sha256"], row["logical_bytes"]) != expected:
        raise RuntimeError(f"record mismatch: {path}")
    return row


def exact_tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {"root": str(root), "inventory": rows, "file_count": len(rows),
            "logical_file_bytes": sum(row[2] for row in rows),
            "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
            "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest()}


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(temporary):
        raise RuntimeError(f"terminal exists: {path}")
    write_exclusive(temporary, cbytes(value))
    os.replace(temporary, path)
    fsync_dir(path.parent)


def directory_identity(path: Path) -> tuple[int, int]:
    observed = os.lstat(path)
    if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
        raise RuntimeError(f"directory identity: {path}")
    return observed.st_dev, observed.st_ino


def cleanup_owned_prep(path: Path, identity: tuple[int, int]) -> None:
    if not os.path.lexists(path) or directory_identity(path) != identity:
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink() or not child.is_file():
            raise RuntimeError(f"unsafe prep cleanup: {child}")
        child.unlink()
    path.rmdir()
    fsync_dir(path.parent)


def close_fsync(stream) -> None:
    if stream is not None and not stream.closed:
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()


def group_empty(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return False
    except ProcessLookupError:
        return True


def terminate(process: subprocess.Popen | None) -> dict:
    result = {"started": process is not None, "term_sent": False, "kill_sent": False,
              "reaped": process is None, "group_empty": True}
    if process is None:
        return result
    if process.poll() is None or not group_empty(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            result["term_sent"] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    if not group_empty(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            result["kill_sent"] = True
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait(timeout=10)
    else:
        process.wait()
    result["reaped"] = process.poll() is not None
    result["group_empty"] = group_empty(process.pid)
    return result


def load_diagnostic():
    spec = importlib.util.spec_from_file_location("v497_diagnostic", DIAGNOSTIC_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("diagnostic import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command() -> list[str]:
    return [str(RLPY), str(DIAGNOSTIC_SOURCE), "--diagnostic-source", str(DIAGNOSTIC_SOURCE),
            "--diagnostic-source-sha", DIAGNOSTIC_SOURCE_SHA, "--output-root", str(DIAGNOSTIC_ROOT)]


def v496_closure() -> dict:
    observed = exact_tree(V496_FAILURE_ROOT)
    if observed != V496_TREE:
        raise RuntimeError("v496 exact6")
    process_record = regular(V496_FAILURE_ROOT / "process_receipt.json", (V496_PROCESS_SHA, 411))
    process = json.loads(Path(process_record["path"]).read_text())
    if (set(process) != {"cleanup", "corrected_inner_wrapper_invocations", "error", "error_type", "format",
                         "materializer_invocations", "outer_wrapper_invocations", "passed", "r2_invocations",
                         "retry_authorized", "status"}
            or process["status"] != "failed_no_retry" or process["passed"] is not False
            or type(process["materializer_invocations"]) is not int or process["materializer_invocations"] != 1
            or type(process["outer_wrapper_invocations"]) is not int or process["outer_wrapper_invocations"] != 0
            or type(process["corrected_inner_wrapper_invocations"]) is not int or process["corrected_inner_wrapper_invocations"] != 0
            or type(process["r2_invocations"]) is not int or process["r2_invocations"] != 0
            or process["retry_authorized"] is not False
            or process["cleanup"].get("reaped") is not True or process["cleanup"].get("group_empty") is not True):
        raise RuntimeError("v496 failure semantics")
    return {"tree": observed, "process_receipt": process_record}


def live_processes() -> list[dict]:
    needles = (str(DIAGNOSTIC_SOURCE), "materialize_v496_v495", "launch_v496_v495",
               "reconcile_v488_v487_c71_exact7_schema_repair.py")
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            text = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(needle in text for needle in needles):
            found.append({"pid": int(entry.name), "cmdline": text})
    return found


def gpu_processes() -> list[str]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("nvidia-smi")
    return [line for line in result.stdout.splitlines() if line.strip()]


def validate_diagnostic(module, source_record: dict) -> tuple[dict, dict, dict]:
    receipt_record = regular(DIAGNOSTIC_ROOT / "diagnostic_receipt.json")
    receipt = json.loads(Path(receipt_record["path"]).read_text())
    if (set(receipt) != RECEIPT_TOP_KEYS or receipt.get("format") != module.FORMAT
            or receipt.get("status") != module.PASS_STATUS or receipt.get("passed") is not True
            or receipt.get("diagnostic_source") != source_record
            or receipt.get("predicate_names") != module.PREDICATE_NAMES
            or receipt.get("all_predicates_passed") is not True
            or receipt.get("snapshots_exactly_equal") is not True
            or receipt.get("snapshot_1") != receipt.get("snapshot_2")
            or not all(row.get("passed") is True for row in receipt.get("predicates", []))
            or receipt.get("runtime_observation") != module.RUNTIME
            or receipt.get("authorization") != module.AUTHORIZATION
            or any(receipt["authorization"].values())):
        raise RuntimeError("diagnostic receipt")
    registration = exact_tree(DIAGNOSTIC_ROOT)
    if registration["inventory"] != [["diagnostic_receipt.json", receipt_record["sha256"], receipt_record["logical_bytes"]]]:
        raise RuntimeError("diagnostic exact1")
    return receipt_record, receipt, registration


def base_process_receipt(status: str, passed: bool, invocations: int, cleanup: dict,
                         error: BaseException | None = None) -> dict:
    row = {"format": "strict-track2-v497-pre-authority-diagnostic-process-receipt-v1",
           "status": status, "passed": passed, "diagnostic_invocations": invocations,
           "authority_materializer_invocations": 0, "outer_wrapper_invocations": 0,
           "corrected_inner_wrapper_invocations": 0, "reconciler_r2_invocations": 0,
           "retry_authorized": False, "cleanup": cleanup}
    if error is not None:
        row.update({"error_type": type(error).__name__, "error": str(error)})
    return row


def commit_terminal(root: Path, receipt: dict, old_mask, terminal_state: dict, phase_hook) -> dict:
    blocked_here = False
    if hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        blocked_here = True
    try:
        phase_hook("before_terminal_write")
        atomic_json(root / "process_receipt.json", receipt)
        observed = exact_tree(root)
        if observed["file_count"] != 6 or [row[0] for row in observed["inventory"]] != EXACT6:
            raise RuntimeError("terminal exact6")
        terminal_state["committed"] = True
        terminal_state["tree"] = observed
        phase_hook("after_terminal_before_unmask")
    finally:
        if blocked_here:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    return terminal_state["tree"]


def orchestrate_once(*, evidence_root: Path, evidence_prep: Path, child_argv: list[str],
                     helper_source: Path, helper_record: dict, transport_script_record: dict,
                     argv_payload: dict, timeout_seconds: float, success_validator, phase_hook=None) -> dict:
    """The sole production and fixture lifecycle: owned prep -> exact5 -> child -> terminal exact6."""
    phase_hook = phase_hook or (lambda _phase: None)
    if os.path.lexists(evidence_root) or os.path.lexists(evidence_prep):
        raise RuntimeError("evidence prestate")
    prep_identity = None
    prep_owned = False
    promoted = False
    process = None
    stdout = None
    stderr = None
    invocations = 0
    terminal_state = {"committed": False, "tree": None}
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, _frame):
        raise ControlledSignal(signum)

    for sig in old_handlers:
        signal.signal(sig, interrupted)
    baseline_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        evidence_prep.mkdir()
        prep_identity = directory_identity(evidence_prep)
        prep_owned = True
        fsync_dir(evidence_prep.parent)
        phase_hook("after_prep_mkdir")
        write_exclusive(evidence_prep / "transport_helper.py", helper_source.read_bytes())
        helper_copy = regular(evidence_prep / "transport_helper.py", (helper_record["sha256"], helper_record["logical_bytes"]))
        argv_payload = {**argv_payload, "transport_script": transport_script_record}
        atomic_json(evidence_prep / "argv.json", argv_payload)
        write_exclusive(evidence_prep / "diagnostic_stdout.log", b"")
        write_exclusive(evidence_prep / "diagnostic_stderr.log", b"")
        atomic_json(evidence_prep / "intent.json", {
            "format": "strict-track2-v497-pre-authority-diagnostic-intent-v1",
            "status": "committed_before_exact_once_diagnostic", "nonce": "fixture" if evidence_root.parent.name.startswith("case-") else os.urandom(32).hex(),
            "argv": regular(evidence_prep / "argv.json"), "retry_authorized": False,
        })
        fsync_dir(evidence_prep)
        if [row[0] for row in exact_tree(evidence_prep)["inventory"]] != EXACT5:
            raise RuntimeError("prep exact5")
        phase_hook("before_promote")
        os.replace(evidence_prep, evidence_root)
        fsync_dir(evidence_root.parent)
        prep_owned = False
        promoted = True
        phase_hook("after_promote_before_unmask")
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)

        stdout = (evidence_root / "diagnostic_stdout.log").open("ab", buffering=0)
        stderr = (evidence_root / "diagnostic_stderr.log").open("ab", buffering=0)
        spawn_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            phase_hook("before_popen")
            process = subprocess.Popen(child_argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, close_fds=True)
            invocations = 1
            phase_hook("after_popen_owned")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)
        returncode = process.wait(timeout=timeout_seconds)
        close_fsync(stdout)
        stdout = None
        close_fsync(stderr)
        stderr = None
        cleanup = terminate(process)
        extra = success_validator(returncode, cleanup)
        result = {**base_process_receipt("passed_exact_once_no_authority_or_reconciliation_execution", True, 1, cleanup),
                  "helper_returncode": 0, "diagnostic_returncode": returncode,
                  "transport_helper": helper_record, "transport_helper_copy": helper_copy,
                  "transport_script": transport_script_record, **extra}
        tree = commit_terminal(evidence_root, result, baseline_mask, terminal_state, phase_hook)
        return {"passed": True, "process_receipt": result, "evidence_tree": tree}
    except BaseException as error:
        if promoted:
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                close_fsync(stdout)
            except BaseException:
                pass
            try:
                close_fsync(stderr)
            except BaseException:
                pass
            cleanup = terminate(process)
            if not terminal_state["committed"]:
                failure = base_process_receipt("failed_no_retry", False, invocations, cleanup, error)
                tree = commit_terminal(evidence_root, failure, baseline_mask, terminal_state, phase_hook)
                return {"passed": False, "process_receipt": failure, "evidence_tree": tree, "error": error}
            return {"passed": bool(json.loads((evidence_root / "process_receipt.json").read_text())["passed"]),
                    "process_receipt": json.loads((evidence_root / "process_receipt.json").read_text()),
                    "evidence_tree": terminal_state["tree"], "deferred_error": error}
        if prep_owned and prep_identity is not None:
            cleanup_owned_prep(evidence_prep, prep_identity)
        raise
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        except BaseException:
            pass
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def production_fixture() -> dict:
    checks = {}
    with tempfile.TemporaryDirectory(prefix="v497-helper-fixture-") as directory:
        base = Path(directory)
        helper = base / "helper.py"
        helper.write_bytes(Path(__file__).read_bytes())
        helper_record = regular(helper)
        script = base / "transport.sh"
        script.write_bytes(b"#!/bin/bash\n")
        script_record = regular(script)

        def run_case(name, code, timeout=5, hook=None, argv_override=None):
            parent = base / ("case-" + name)
            parent.mkdir()
            child = parent / "child.py"
            child.write_text(code)
            root = parent / "evidence"
            prep = parent / "evidence.execution-prep"
            argv = argv_override or [sys.executable, str(child)]
            payload = {"format": "fixture", "argv": argv}
            return orchestrate_once(evidence_root=root, evidence_prep=prep, child_argv=argv,
                                    helper_source=helper, helper_record=helper_record,
                                    transport_script_record=script_record, argv_payload=payload,
                                    timeout_seconds=timeout,
                                    success_validator=lambda rc, clean: ({"fixture": True} if rc == 0 and clean["reaped"] and clean["group_empty"] else (_ for _ in ()).throw(RuntimeError(f"child rc {rc}"))),
                                    phase_hook=hook), root, prep

        result, root, prep = run_case("success", "print('ok')\n")
        checks["success_exact6"] = result["passed"] and result["evidence_tree"]["file_count"] == 6 and not os.path.lexists(prep)
        result, root, _ = run_case("nonzero", "raise SystemExit(7)\n")
        checks["nonzero_failure_exact6"] = not result["passed"] and result["process_receipt"]["status"] == "failed_no_retry" and result["evidence_tree"]["file_count"] == 6
        result, root, _ = run_case("popen", "pass\n", argv_override=[str(base / "missing-executable")])
        checks["popen_error_exact6"] = not result["passed"] and result["process_receipt"]["diagnostic_invocations"] == 0 and result["evidence_tree"]["file_count"] == 6
        result, root, _ = run_case("timeout", "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n", timeout=0.2)
        checks["timeout_kill_reap_exact6"] = (not result["passed"] and result["process_receipt"]["cleanup"]["kill_sent"]
                                                   and result["process_receipt"]["cleanup"]["reaped"]
                                                   and result["process_receipt"]["cleanup"]["group_empty"])
        def pending_hook(phase):
            if phase == "after_promote_before_unmask":
                os.kill(os.getpid(), signal.SIGTERM)
        result, root, _ = run_case("signal", "print('never')\n", hook=pending_hook)
        checks["pending_signal_failure_exact6"] = not result["passed"] and result["process_receipt"]["diagnostic_invocations"] == 0 and result["evidence_tree"]["file_count"] == 6
        def preintent_hook(phase):
            if phase == "after_prep_mkdir":
                raise RuntimeError("fixture preintent")
        try:
            run_case("preintent", "print('never')\n", hook=preintent_hook)
            checks["preintent_owned_cleanup"] = False
        except RuntimeError:
            parent = base / "case-preintent"
            checks["preintent_owned_cleanup"] = not os.path.lexists(parent / "evidence") and not os.path.lexists(parent / "evidence.execution-prep")
    result = {"passed": all(checks.values()), "checks": checks, "check_count": len(checks)}
    result["evidence_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        result = production_fixture()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["passed"] else 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--helper-sha", required=True)
    parser.add_argument("--helper-bytes", type=int, required=True)
    parser.add_argument("--transport-script", type=Path, required=True)
    parser.add_argument("--transport-script-sha", required=True)
    parser.add_argument("--transport-script-bytes", type=int, required=True)
    args = parser.parse_args()
    if Path(sys.executable) != RLPY:
        raise RuntimeError("interpreter")
    if args.helper_source != SELF_PATH or args.helper_source.resolve() != SELF_PATH or Path(__file__).resolve() != SELF_PATH:
        raise RuntimeError("helper canonical")
    if args.transport_script != TRANSPORT_SCRIPT_PATH or args.transport_script.resolve() != TRANSPORT_SCRIPT_PATH:
        raise RuntimeError("transport script canonical")
    helper_record = regular(SELF_PATH, (args.helper_sha, args.helper_bytes))
    script_record = regular(TRANSPORT_SCRIPT_PATH, (args.transport_script_sha, args.transport_script_bytes))
    source_record = regular(DIAGNOSTIC_SOURCE, (DIAGNOSTIC_SOURCE_SHA, DIAGNOSTIC_SOURCE_BYTES))
    module = load_diagnostic()
    if module.SELF_PATH != DIAGNOSTIC_SOURCE or module.DIAGNOSTIC_ROOT != DIAGNOSTIC_ROOT:
        raise RuntimeError("diagnostic boundary")
    pre = {"diagnostic_inputs": module.immutable_input_snapshot(source_record),
           "v496_failure": v496_closure(), "transport_helper": helper_record,
           "transport_script": script_record}
    if (os.path.lexists(DIAGNOSTIC_ROOT) or os.path.lexists(DIAGNOSTIC_PREP)
            or os.path.lexists(EVIDENCE_ROOT) or os.path.lexists(EVIDENCE_PREP)
            or live_processes() or gpu_processes()):
        raise RuntimeError("prestate")
    argv = command()
    argv_payload = {"format": "strict-track2-v497-pre-authority-diagnostic-argv-v1",
                    "argv": argv, "argv_repr": [repr(token) for token in argv],
                    "argv_utf8_hex": [token.encode().hex() for token in argv],
                    "all_tokens_no_cr_lf": all("\r" not in token and "\n" not in token for token in argv),
                    "diagnostic_invocations_before": 0, "authority_materializer_invocations": 0,
                    "outer_wrapper_invocations": 0, "corrected_inner_wrapper_invocations": 0,
                    "reconciler_r2_invocations": 0, "pre_snapshot": pre}
    started = time.monotonic()

    def validate_success(returncode, cleanup):
        if returncode or not cleanup["reaped"] or not cleanup["group_empty"]:
            raise RuntimeError(f"diagnostic rc {returncode}")
        receipt_record, _, registration = validate_diagnostic(module, source_record)
        post = {"diagnostic_inputs": module.immutable_input_snapshot(source_record),
                "v496_failure": v496_closure(), "transport_helper": helper_record,
                "transport_script": script_record}
        if post != pre or os.path.lexists(DIAGNOSTIC_PREP) or live_processes() or gpu_processes():
            raise RuntimeError("poststate")
        wall = time.monotonic() - started
        if not math.isfinite(wall) or wall < 0:
            raise RuntimeError("wall")
        return {"wall_seconds": wall, "argv": regular(EVIDENCE_ROOT / "argv.json"),
                "intent": regular(EVIDENCE_ROOT / "intent.json"),
                "stdout": regular(EVIDENCE_ROOT / "diagnostic_stdout.log"),
                "stderr": regular(EVIDENCE_ROOT / "diagnostic_stderr.log"),
                "pre_snapshot": pre, "post_snapshot": post,
                "pre_post_snapshots_exactly_equal": True,
                "diagnostic_receipt": receipt_record, "diagnostic_registration_tree": registration}

    result = orchestrate_once(evidence_root=EVIDENCE_ROOT, evidence_prep=EVIDENCE_PREP,
                              child_argv=argv, helper_source=SELF_PATH, helper_record=helper_record,
                              transport_script_record=script_record, argv_payload=argv_payload,
                              timeout_seconds=300, success_validator=validate_success)
    print(json.dumps(result, sort_keys=True, default=str), flush=True)
    if not result["passed"]:
        raise RuntimeError(str(result["process_receipt"].get("error", "diagnostic failure")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
