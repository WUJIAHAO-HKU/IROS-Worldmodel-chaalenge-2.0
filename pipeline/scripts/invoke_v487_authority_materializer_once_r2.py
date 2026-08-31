import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RLPY = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
CONTRACT = ROOT / "pipeline/scripts/v487_v486_phase_a_static_reconciliation_postregistration_authority_contract.json"
MATERIALIZER = ROOT / "pipeline/scripts/materialize_v487_v486_phase_a_static_reconciliation_postregistration_authority.py"
WRAPPER = ROOT / "pipeline/scripts/launch_v487_v486_v485_static_false_positive_reconciliation.py"
SELF_PATH = ROOT / "pipeline/scripts/invoke_v487_authority_materializer_once_r2.py"
AUTHORITY_ROOT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v487_v486_phase_a_static_reconciliation_authority_seed1629_20260824"
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")
ATTEMPT_ROOT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v487_v486_phase_a_static_reconciliation_attempt_seed1629_20260824"
F813_ROOT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v486_v485_phase_a_static_reconciliation_seed1628_20260824"
TRANSPARENT = F813_ROOT / "transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")
OLD_EVIDENCE_ROOT = ROOT / "pipeline/scripts/v487_v486_phase_a_static_reconciliation_authority_materialization_evidence_20260824"
NEW_EVIDENCE_ROOT = ROOT / "pipeline/scripts/v487_v486_phase_a_static_reconciliation_authority_materialization_evidence_r2_20260824"

CONTRACT_SHA = "a1c68466a89826a64336b3231efa795ab6c43b8da633883743e58ef03a7509ea"
CONTRACT_BYTES = 16830
MATERIALIZER_SHA = "6c7b5e259e388d59bfb71a09c068472037dbde74468f5d9ba1b96fa92a8abd3d"
MATERIALIZER_BYTES = 18312
WRAPPER_SHA = "41a0d93428d1b4ab224bbc3d7bfd8e4e3752522a3ea87de2439fbb57ed7d003b"
WRAPPER_BYTES = 63286

OLD_EVIDENCE_TREE = {
    "inventory": [
        ["invoke_v487_authority_materializer_once.py", "8debd9ab4b939aa56deee3db994b14b75b074080a6ca105a6bd2a25d6201994d", 4523],
        ["materializer_argv.json", "69dac7cc4fe4d5677c8e331090f96aa12c4f613eaff5f46e9bade8cf0bfdb8fe", 5256],
        ["materializer_stderr.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["materializer_stdout.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
    ],
    "file_count": 4,
    "logical_file_bytes": 9779,
    "sha256sum_lines_digest_sha256": "12b38c78146c6638a8b449127f33f83350d5a66b71777311bf0bc3785ebc9442",
    "canonical_json_triples_digest_sha256": "71835d08525e3be53a2e5bebbdb5aa97921257a70b84db890c5ee32e94730ca2",
}
F813_TREE = {
    "inventory": [
        ["immutable_evidence/v485_static_b73.log", "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b", 6475],
        ["preregistration.json", "f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8", 21296],
    ],
    "file_count": 2,
    "logical_file_bytes": 27771,
    "sha256sum_lines_digest_sha256": "717d6ae12fbacbaafa147d03503140e5f807c028acc97a7294c84da2d79fe3fe",
    "canonical_json_triples_digest_sha256": "a0602666a7e59d60e464b3bc76b31b2fbf501e3d37bbd0b18b2db8c1233971ec",
}
CURRENT10_NAMES = {
    "qualification_root",
    "old_phase_a_attempts",
    "old_phase_a_authority",
    "transparent_output",
    "transparent_tmp",
    "superseded_v486_authority_root",
    "superseded_v486_authority_prep",
    "superseded_v486_attempt_root",
    "fresh_v487_authority_prep",
    "fresh_v487_attempt_root",
}


class ControlledSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"controlled signal {signum}")
        self.signum = signum


def controlled_signal_handler(signum: int, _frame: object) -> None:
    raise ControlledSignal(signum)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def file_record(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict:
    if path.is_symlink() or not path.is_file() or path != path.resolve():
        raise RuntimeError(f"regular canonical file: {path}")
    data = path.read_bytes()
    result = {"path": str(path), "sha256": digest_bytes(data), "logical_bytes": len(data)}
    if expected_sha is not None and result["sha256"] != expected_sha:
        raise RuntimeError(f"sha drift: {path}")
    if expected_bytes is not None and result["logical_bytes"] != expected_bytes:
        raise RuntimeError(f"bytes drift: {path}")
    return result


def tree(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir() or root != root.resolve():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            data = path.read_bytes()
            rows.append([path.relative_to(root).as_posix(), digest_bytes(data), len(data)])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": digest_bytes(lines),
        "canonical_json_triples_digest_sha256": digest_bytes(triples),
    }


def fsync_dir(path: Path) -> None:
    if os.name == "nt":  # synthetic fixture only; production execution is Linux.
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):
        raise RuntimeError(f"one-shot path exists: {path}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError(f"short atomic write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, canonical_bytes(value))


def strict_canonical_readback(path: Path, expected: object) -> object:
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"canonical JSON parse: {path}") from exc
    if parsed != expected or raw != canonical_bytes(parsed):
        raise RuntimeError(f"canonical JSON exact readback: {path}")
    return parsed


def source_records(self_sha: str, self_bytes: int) -> list[dict]:
    return [
        file_record(CONTRACT, CONTRACT_SHA, CONTRACT_BYTES),
        file_record(MATERIALIZER, MATERIALIZER_SHA, MATERIALIZER_BYTES),
        file_record(WRAPPER, WRAPPER_SHA, WRAPPER_BYTES),
        file_record(SELF_PATH, self_sha, self_bytes),
    ]


def current10_absence_snapshot() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    specs = contract.get("required_current_absences_after_authority")
    if not isinstance(specs, dict) or set(specs) != CURRENT10_NAMES:
        raise RuntimeError("contract current10 names")
    result = {}
    for name, spec in sorted(specs.items()):
        if not isinstance(spec, dict) or set(spec) != {"kind", "path"} or spec["kind"] not in {"file", "directory"}:
            raise RuntimeError(f"contract current10 schema: {name}")
        path = Path(spec["path"])
        if path != path.resolve():
            raise RuntimeError(f"contract current10 path: {name}")
        result[name] = {**spec, "absent": not os.path.lexists(path)}
    return result


def closure_snapshot(self_sha: str, self_bytes: int) -> dict:
    sources = source_records(self_sha, self_bytes)
    old_tree = tree(OLD_EVIDENCE_ROOT)
    if old_tree != OLD_EVIDENCE_TREE:
        raise RuntimeError("old transport evidence drift")
    old_argv = OLD_EVIDENCE_ROOT / "materializer_argv.json"
    old_raw = old_argv.read_bytes()
    if not old_raw.endswith(b"\\n") or old_raw.endswith(b"\n"):
        raise RuntimeError("old malformed tail fact")
    try:
        json.loads(old_raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data" or exc.pos != 5254:
            raise RuntimeError("old malformed parse fact") from exc
    else:
        raise RuntimeError("old argv unexpectedly parseable")
    if os.path.lexists(OLD_EVIDENCE_ROOT / "materializer_invocation_intent.json") or os.path.lexists(
        OLD_EVIDENCE_ROOT / "materializer_process_receipt.json"
    ):
        raise RuntimeError("old helper invocation evidence changed")
    if tree(F813_ROOT) != F813_TREE:
        raise RuntimeError("f813 tree drift")
    current10 = current10_absence_snapshot()
    if not all(item["absent"] for item in current10.values()):
        raise RuntimeError("current10 absence")
    return {
        "source_records": sources,
        "old_transport_evidence_tree": old_tree,
        "old_argv_malformed_literal_backslash_n": True,
        "old_helper_failed_before_intent": True,
        "old_materializer_invocations": 0,
        "old_invocation_intent_absent": True,
        "old_process_receipt_absent": True,
        "f813_tree": F813_TREE,
        "current10_absences": current10,
    }


def verify_preflight(self_sha: str, self_bytes: int) -> dict:
    if os.path.lexists(NEW_EVIDENCE_ROOT):
        raise RuntimeError("fresh evidence root already exists")
    if os.path.lexists(AUTHORITY_ROOT):
        raise RuntimeError("fresh authority root already exists")
    return closure_snapshot(self_sha, self_bytes)


def materializer_argv() -> list[str]:
    return [
        str(RLPY),
        str(MATERIALIZER),
        "--contract",
        str(CONTRACT),
        "--contract-sha",
        CONTRACT_SHA,
        "--materializer-source",
        str(MATERIALIZER),
        "--materializer-sha",
        MATERIALIZER_SHA,
        "--authority-root",
        str(AUTHORITY_ROOT),
    ]


def process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def terminate_and_reap(process: subprocess.Popen) -> bool:
    pid = process.pid
    if process.poll() is not None:
        process.wait()
        if process_group_gone(pid):
            return True
    for sig, wait_seconds in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            process.wait()
            return process_group_gone(pid)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if process_group_gone(pid):
                if process.poll() is None:
                    process.wait()
                return True
            if process.poll() is not None:
                time.sleep(0.05)
                continue
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
            if process_group_gone(pid):
                return True
            break
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return False
    return process_group_gone(pid)


def poststate() -> dict:
    result = {"current10_absences": current10_absence_snapshot(), "authority_tree": None, "authority_receipt": None}
    if AUTHORITY_ROOT.is_dir() and not AUTHORITY_ROOT.is_symlink():
        result["authority_tree"] = tree(AUTHORITY_ROOT)
        receipt = AUTHORITY_ROOT / "authority_receipt.json"
        if result["authority_tree"]["file_count"] == 1 and receipt.is_file() and not receipt.is_symlink():
            result["authority_receipt"] = file_record(receipt)
    return result


def cleanup_preintent_root(created_by_self: bool) -> None:
    if not created_by_self or not os.path.lexists(NEW_EVIDENCE_ROOT):
        return
    allowed = {
        "materializer_argv.json",
        "materializer_argv.json.tmp",
        "materializer_stdout.log",
        "materializer_stdout.log.tmp",
        "materializer_stderr.log",
        "materializer_stderr.log.tmp",
        "materializer_invocation_intent.json.tmp",
    }
    entries = list(NEW_EVIDENCE_ROOT.iterdir())
    if any(entry.name not in allowed or entry.is_dir() or entry.is_symlink() for entry in entries):
        raise RuntimeError("unsafe pre-intent cleanup")
    for entry in entries:
        entry.unlink()
    NEW_EVIDENCE_ROOT.rmdir()
    fsync_dir(NEW_EVIDENCE_ROOT.parent)


def execute(self_sha: str, self_bytes: int) -> int:
    preflight = verify_preflight(self_sha, self_bytes)
    preflight_digest = digest_bytes(canonical_bytes(preflight))
    argv_path = NEW_EVIDENCE_ROOT / "materializer_argv.json"
    stdout_path = NEW_EVIDENCE_ROOT / "materializer_stdout.log"
    stderr_path = NEW_EVIDENCE_ROOT / "materializer_stderr.log"
    intent_path = NEW_EVIDENCE_ROOT / "materializer_invocation_intent.json"
    receipt_path = NEW_EVIDENCE_ROOT / "materializer_process_receipt.json"
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    created_by_self = False
    intent_committed = False
    terminal_committed = False
    process = None
    stdout = None
    stderr = None
    started = None
    timed_out = False
    launch_error = None
    materializer_returncode = None
    invocations = 0
    group_empty = True

    def close_and_fsync_logs() -> None:
        nonlocal stdout, stderr
        for stream in (stdout, stderr):
            if stream is not None and not stream.closed:
                os.fsync(stream.fileno())
                stream.close()

    def error_record(exc: BaseException) -> dict:
        value = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, ControlledSignal):
            value["signal"] = exc.signum
        return value

    for sig in old_handlers:
        signal.signal(sig, controlled_signal_handler)
    try:
        create_mask = None
        try:
            if hasattr(signal, "pthread_sigmask"):
                create_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            NEW_EVIDENCE_ROOT.mkdir(mode=0o755)
            created_by_self = True
        finally:
            if create_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, create_mask)
        fsync_dir(NEW_EVIDENCE_ROOT.parent)
        argv = materializer_argv()
        argv_evidence = {
            "format": "strict-track2-v487-authority-materializer-r2-argv-evidence-v1",
            "argv": argv,
            "argv_repr": [repr(token) for token in argv],
            "argv_utf8_hex": [token.encode("utf-8").hex() for token in argv],
            "all_tokens_no_cr_lf": all("\r" not in token and "\n" not in token for token in argv),
            "preflight": preflight,
            "preflight_digest_sha256": preflight_digest,
        }
        atomic_json(argv_path, argv_evidence)
        strict_canonical_readback(argv_path, argv_evidence)
        if argv_evidence["argv"] != materializer_argv() or argv_evidence["all_tokens_no_cr_lf"] is not True:
            raise RuntimeError("argv drift before intent")
        atomic_bytes(stdout_path, b"")
        atomic_bytes(stderr_path, b"")
        intent = {
            "format": "strict-track2-v487-authority-materializer-r2-invocation-intent-v1",
            "argv_evidence": file_record(argv_path),
            "transport_helper": file_record(SELF_PATH, self_sha, self_bytes),
            "materializer_invocations_before": 0,
            "preflight_digest_sha256": preflight_digest,
            "created_unix_ns": time.time_ns(),
        }
        intent_mask = None
        try:
            if hasattr(signal, "pthread_sigmask"):
                intent_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            atomic_json(intent_path, intent)
            intent_committed = True
        finally:
            if intent_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, intent_mask)

        stdout = stdout_path.open("ab", buffering=0)
        stderr = stderr_path.open("ab", buffering=0)
        started = time.monotonic()
        old_mask = None
        try:
            if hasattr(signal, "pthread_sigmask"):
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            invocations = 1
        finally:
            if old_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        try:
            materializer_returncode = int(process.wait(timeout=300))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_and_reap(process)
            materializer_returncode = 124
        group_empty = terminate_and_reap(process)
        close_and_fsync_logs()

        post_closure = closure_snapshot(self_sha, self_bytes)
        post_closure_digest = digest_bytes(canonical_bytes(post_closure))
        closures_equal = post_closure == preflight and post_closure_digest == preflight_digest
        state = poststate()
        validation_error = None
        if materializer_returncode == 0:
            try:
                if (
                    not closures_equal
                    or state["authority_tree"] is None
                    or state["authority_tree"]["file_count"] != 1
                    or state["authority_tree"]["inventory"][0][0] != "authority_receipt.json"
                    or state["authority_receipt"] is None
                    or not all(item["absent"] for item in state["current10_absences"].values())
                    or not group_empty
                ):
                    raise RuntimeError("successful poststate")
            except Exception as exc:
                validation_error = error_record(exc)
        helper_returncode = 0 if materializer_returncode == 0 and validation_error is None else (
            materializer_returncode if materializer_returncode not in (None, 0) else 125
        )
        receipt = {
            "format": "strict-track2-v487-authority-materializer-r2-process-receipt-v1",
            "status": "passed" if helper_returncode == 0 else "failed",
            "helper_returncode": helper_returncode,
            "materializer_returncode": materializer_returncode,
            "materializer_invocations": invocations,
            "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started,
            "process_group_empty": group_empty,
            "launch_error": launch_error,
            "validation_error": validation_error,
            "argv_evidence": file_record(argv_path),
            "invocation_intent": file_record(intent_path),
            "stdout": file_record(stdout_path),
            "stderr": file_record(stderr_path),
            "transport_helper": file_record(SELF_PATH, self_sha, self_bytes),
            "pre_closure": preflight,
            "post_closure": post_closure,
            "pre_closure_digest_sha256": preflight_digest,
            "post_closure_digest_sha256": post_closure_digest,
            "pre_post_closure_exactly_equal": closures_equal,
            "poststate": state,
        }
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        atomic_json(receipt_path, receipt)
        terminal_committed = True
        strict_canonical_readback(receipt_path, receipt)
        print(json.dumps({"receipt": file_record(receipt_path), "returncode": helper_returncode}, sort_keys=True), flush=True)
        return helper_returncode
    except BaseException as exc:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        intent_committed = intent_committed or os.path.lexists(intent_path)
        terminal_committed = terminal_committed or os.path.lexists(receipt_path)
        if process is not None:
            group_empty = terminate_and_reap(process)
            if process.returncode is not None:
                materializer_returncode = int(process.returncode)
        close_and_fsync_logs()
        if terminal_committed:
            return 128 + exc.signum if isinstance(exc, ControlledSignal) else 125
        if intent_committed:
            failure_error = error_record(exc)
            try:
                post_closure = closure_snapshot(self_sha, self_bytes)
                post_closure_digest = digest_bytes(canonical_bytes(post_closure))
                closures_equal = post_closure == preflight and post_closure_digest == preflight_digest
            except BaseException as closure_exc:
                post_closure = {"inspection_error": error_record(closure_exc)}
                post_closure_digest = None
                closures_equal = False
            try:
                state = poststate()
            except BaseException as state_exc:
                state = {"inspection_error": error_record(state_exc)}
            helper_returncode = 128 + exc.signum if isinstance(exc, ControlledSignal) else 125
            failure_receipt = {
                "format": "strict-track2-v487-authority-materializer-r2-process-receipt-v1",
                "status": "failed",
                "helper_returncode": helper_returncode,
                "materializer_returncode": materializer_returncode,
                "materializer_invocations": invocations,
                "timed_out": timed_out,
                "wall_seconds": 0.0 if started is None else time.monotonic() - started,
                "process_group_empty": group_empty,
                "launch_error": launch_error,
                "validation_error": failure_error,
                "argv_evidence": file_record(argv_path),
                "invocation_intent": file_record(intent_path),
                "stdout": file_record(stdout_path),
                "stderr": file_record(stderr_path),
                "transport_helper": file_record(SELF_PATH, self_sha, self_bytes),
                "pre_closure": preflight,
                "post_closure": post_closure,
                "pre_closure_digest_sha256": preflight_digest,
                "post_closure_digest_sha256": post_closure_digest,
                "pre_post_closure_exactly_equal": closures_equal,
                "poststate": state,
            }
            atomic_json(receipt_path, failure_receipt)
            terminal_committed = True
            strict_canonical_readback(receipt_path, failure_receipt)
            return helper_returncode
        cleanup_preintent_root(created_by_self)
        return 128 + exc.signum if isinstance(exc, ControlledSignal) else 125
    finally:
        close_and_fsync_logs()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def synthetic_self_test() -> int:
    checks = {}
    with tempfile.TemporaryDirectory(prefix="v487_helper_r2_") as td:
        root = Path(td)
        canonical = root / "canonical.json"
        expected = {"argv": ["a", "b c"], "sha": "0" * 64}
        atomic_json(canonical, expected)
        checks["canonical_dump_readback_exact"] = strict_canonical_readback(canonical, expected) == expected
        checks["canonical_ends_single_lf"] = canonical.read_bytes().endswith(b"\n") and not canonical.read_bytes().endswith(b"\\n")

        malformed = root / "malformed.json"
        malformed.write_bytes(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode() + b"\\n")
        rejected = False
        try:
            strict_canonical_readback(malformed, expected)
        except RuntimeError:
            rejected = True
        checks["literal_backslash_n_rejected"] = rejected

        zero = root / "zero"
        zero.mkdir()
        checks["zero_state_has_no_intent_receipt_or_logs"] = all(
            not os.path.lexists(zero / name)
            for name in (
                "materializer_invocation_intent.json",
                "materializer_process_receipt.json",
                "materializer_stdout.log",
                "materializer_stderr.log",
            )
        )
        argv = materializer_argv()
        checks["argv_exact_and_crlf_free"] = len(argv) == 12 and all("\r" not in token and "\n" not in token for token in argv)
        checks["old_exact4_tree_constants"] = (
            OLD_EVIDENCE_TREE["file_count"] == 4
            and OLD_EVIDENCE_TREE["logical_file_bytes"] == 9779
            and len(OLD_EVIDENCE_TREE["inventory"]) == 4
        )
        checks["contract_current10_names_frozen"] = len(CURRENT10_NAMES) == 10 and {
            "superseded_v486_authority_root",
            "superseded_v486_authority_prep",
            "superseded_v486_attempt_root",
            "fresh_v487_authority_prep",
            "fresh_v487_attempt_root",
        }.issubset(CURRENT10_NAMES)
        caught_signal = None
        prior_handler = signal.signal(signal.SIGTERM, controlled_signal_handler)
        try:
            signal.raise_signal(signal.SIGTERM)
        except ControlledSignal as exc:
            caught_signal = exc
        finally:
            signal.signal(signal.SIGTERM, prior_handler)
        checks["real_signal_converted_to_controlled_baseexception"] = (
            isinstance(caught_signal, ControlledSignal)
            and isinstance(caught_signal, BaseException)
            and not isinstance(caught_signal, Exception)
            and caught_signal.signum == signal.SIGTERM
        )
        if os.name != "nt":
            ready = root / "signal_child_ready"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)",
                    str(ready),
                ],
                start_new_session=True,
            )
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            checks["signal_cleanup_term_kill_wait_reap_group_empty"] = (
                ready.exists() and terminate_and_reap(child) and child.poll() is not None
            )
        else:
            checks["signal_cleanup_term_kill_wait_reap_group_empty"] = True
        sample_closure = {"old": OLD_EVIDENCE_TREE, "current10": sorted(CURRENT10_NAMES)}
        checks["pre_post_closure_digest_exact_gate"] = digest_bytes(canonical_bytes(sample_closure)) == digest_bytes(
            canonical_bytes(json.loads(canonical_bytes(sample_closure)))
        )
    passed = all(checks.values())
    print(json.dumps({"checks": checks, "passed": passed}, sort_keys=True))
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha", required=True)
    parser.add_argument("--self-bytes", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    args = parse_args()
    if len(args.self_sha) != 64 or any(char not in "0123456789abcdef" for char in args.self_sha) or args.self_bytes <= 0:
        raise RuntimeError("self record args")
    return execute(args.self_sha, args.self_bytes)


if __name__ == "__main__":
    sys.exit(main())
