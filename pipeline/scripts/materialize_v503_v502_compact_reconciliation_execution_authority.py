#!/usr/bin/env python3
"""Materialize only the compact v503 reconciliation execution authority."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import stat
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CONTRACT_PATH = ROOT / "pipeline/scripts/v503_v502_compact_reconciliation_execution_authority_contract.json"
MATERIALIZER_PATH = ROOT / "pipeline/scripts/materialize_v503_v502_compact_reconciliation_execution_authority.py"
WRAPPER_PATH = ROOT / "pipeline/scripts/launch_v503_v502_compact_single_reconciler.py"
R2_PATH = ROOT / "pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py"
PHASE_PATH = ROOT / "pipeline/scripts/v485_v482_v169_cache_determinism_scope_repair_contract.json"
REPAIR_PATH = J / "v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json"
STATIC_PATH = J / "v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json"
F813_ROOT = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824"
V502_AUTH_ROOT = J / "v502_v501_inner_binding_path_repair_execution_authority_seed1644_20260825"
V502_EVIDENCE_ROOT = J / "v502_v501_inner_binding_path_repair_execution_authority_materialization_evidence_seed1644_20260825"
FAILURE_PATH = ROOT / "pipeline/scripts/v503_v502_outer_preintent_v494_ancestry_failure_forensic_reconstructed.json"
V504_FAILURE_PATH = ROOT / "pipeline/scripts/v504_v503_v502_v485_alias_restore_pre_evidence_failure_forensic_reconstructed.json"
V504_DESIGN_PATH = ROOT / "pipeline/scripts/v504_v503_v502_v485_volatile_static_log_alias_restore_design.json"
V504_HELPER_PATH = Path("/root/restore_v504_v503_v485_volatile_static_log_alias_once.py")
V504_SCRIPT_PATH = Path("/root/v504_restore_v485_volatile_static_log_alias_once.sh")
V504_EVIDENCE_ROOT = J / "v504_v503_v502_v485_volatile_static_log_alias_restore_evidence_seed1646_20260825"
V504_TARGET_PATH = Path("/dev/shm/v485_static_b73.log")
AUTHORITY_ROOT = J / "v503_v502_compact_reconciliation_execution_authority_seed1645_20260825"
WRAPPER_ATTEMPT_ROOT = J / "v503_v502_compact_single_reconciler_attempt_seed1645_20260825"
TRANSPARENT_PATH = F813_ROOT / "transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

CONTRACT_FORMAT = "strict-track2-v503-v502-compact-reconciliation-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v503-v502-compact-reconciliation-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_compact_single_reconciler_attempt"

CONTRACT_SOURCE_ORDER = [
    "authority_materializer", "direct_reconciler_wrapper", "reconciler_r2",
    "phase_a_design_contract", "repair_formal_receipt", "postregistration_static_receipt",
]
AUTHORITY_SOURCE_ORDER = ["authority_design_contract", *CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES = {
    "authority_design_contract": "authority_design_contract",
    "authority_materializer": "authority_materializer_source",
    "direct_reconciler_wrapper": "direct_reconciler_wrapper_source",
    "reconciler_r2": "reconciler_r2_source",
    "phase_a_design_contract": "phase_a_design_contract_source",
    "repair_formal_receipt": "repair_formal_source",
    "postregistration_static_receipt": "postregistration_static_source",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_role_order",
    "source_aliases", "source_closure_sha256", "authority_materializer_source",
    "repair_formal_record", "postregistration_static_record", "phase_a_design_contract_record",
    "f813_registration_tree", "v502_authority_receipt", "v502_authority_registration_tree",
    "v502_authority_materialization_evidence_tree", "v502_authority_materializer_process_receipt",
    "v502_failure_summary", "v502_failure_ancestry", "historical_absences",
    "v504_restore_failure_forensic", "v504_restore_design", "v504_restore_helper",
    "v504_restore_script", "v504_restore_evidence_tree", "v504_restore_process_receipt",
    "v504_restored_target", "v504_restore_ancestry",
    "current_absences_after_authority", "authority_receipt_contract", "authorization",
    "runtime_observation", "execution_boundary",
}
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
    "rl_authorized": False,
    "submission_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME = {
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
CHECK_KEYS = sorted({
    "authority_contract_current", "authority_materializer_current", "direct_wrapper_current",
    "r2_current", "phase_contract_current", "repair_formal_current", "static_receipt_current",
    "source_closure_current", "source_order_exact", "source_aliases_exact", "f813_exact2",
    "v502_authority_exact1", "v502_evidence_exact6", "v502_process_exact",
    "v502_failure_summary_exact", "v502_failure_leafdiff_exact2", "v502_no_retry_partition",
    "historical_absences", "current_absences", "input_snapshots_equal", "no_live_process",
    "gpu_empty", "authorization_boundary", "execution_boundary", "no_pending_values",
    "v504_failure_forensic_exact", "v504_restore_design_exact", "v504_restore_sources_exact",
    "v504_restore_evidence_exact6", "v504_restore_process_exact", "v504_target_current",
    "v504_restore_partition", "v504_restore_snapshots_equal",
})
AUTH_TOP_KEYS = {
    "format", "status", "passed", "source_closure", "source_role_order", "source_aliases",
    "source_closure_sha256", "authority_design_contract", "authority_materializer_source",
    "direct_reconciler_wrapper_source", "reconciler_r2_source", "phase_a_design_contract_source",
    "repair_formal_source", "postregistration_static_source", "repair_formal_record",
    "postregistration_static_record", "phase_a_design_contract_record", "f813_registration_tree",
    "v502_authority_receipt", "v502_authority_registration_tree",
    "v502_authority_materialization_evidence_tree", "v502_authority_materializer_process_receipt",
    "v502_failure_summary", "v502_failure_ancestry", "wrapper_attempt_root",
    "v504_restore_failure_forensic", "v504_restore_design", "v504_restore_helper",
    "v504_restore_script", "v504_restore_evidence_tree", "v504_restore_process_receipt",
    "v504_restored_target", "v504_restore_ancestry",
    "transparent_static_receipt_path", "historical_absences", "required_absences", "checks",
    "check_keys", "check_key_set_sha256", "checks_sha256", "input_pre_snapshot",
    "input_post_snapshot", "input_snapshots_exactly_equal", "authorization",
    "runtime_observation", "execution_boundary",
}

TERMINAL_COMMITTED = False


class ControlledSignal(BaseException):
    pass


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pending(value) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(pending(item) for item in value.values())
    if isinstance(value, list):
        return any(pending(item) for item in value)
    return False


def regular(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict:
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError("regular: " + str(path))
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError("not regular: " + str(path))
    row = {"path": str(path), "sha256": sha(path), "logical_bytes": st.st_size}
    if expected_sha is not None and row["sha256"] != expected_sha:
        raise RuntimeError("sha: " + str(path))
    if expected_bytes is not None and row["logical_bytes"] != expected_bytes:
        raise RuntimeError("bytes: " + str(path))
    return row


def record(value) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record schema")
    return regular(Path(value["path"]), value["sha256"], value["logical_bytes"])


def tree(root: Path) -> dict:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("tree symlink")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError("tree special")
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in rows).encode()
    canon = json.dumps(rows, separators=(",", ":")).encode()
    return {"root": str(root), "inventory": rows, "file_count": len(rows),
            "logical_file_bytes": sum(row[2] for row in rows),
            "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
            "canonical_json_triples_digest_sha256": hashlib.sha256(canon).hexdigest()}


def verify_tree(value) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "root", "inventory", "file_count", "logical_file_bytes",
        "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256",
    }:
        raise RuntimeError("tree schema")
    observed = tree(Path(value["root"]))
    if observed != value:
        raise RuntimeError("tree mismatch")
    return observed


def decode_absences(value: dict, keys: set[str]) -> dict[str, Path]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("absence keys")
    result = {}
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != {"path"}:
            raise RuntimeError("absence row")
        path = Path(row["path"])
        if path != path.resolve():
            raise RuntimeError("absence canonical")
        result[name] = path
    return result


def snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, Path]) -> dict:
    value = {
        "files": {name: regular(Path(path)) for name, path in sorted(files.items())},
        "trees": {name: tree(Path(path)) for name, path in sorted(trees.items())},
        "absences": {name: {"path": str(path), "absent": not os.path.lexists(path)}
                     for name, path in sorted(absences.items())},
    }
    if not all(row["absent"] for row in value["absences"].values()):
        raise RuntimeError("absence present")
    value["snapshot_sha256"] = csha(value)
    return value


def live_processes(fragments: set[str]) -> list[dict]:
    own = {os.getpid(), os.getppid()}
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if any(fragment in cmd for fragment in fragments):
            rows.append({"pid": int(entry.name), "cmdline": cmd})
    return rows


def gpu_empty() -> bool:
    import subprocess
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            text=True, capture_output=True, check=False)
    return result.returncode == 0 and not result.stdout.strip()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise RuntimeError("short write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


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


def signal_handler(signum, _frame):
    if TERMINAL_COMMITTED:
        return
    raise ControlledSignal(signum)


def publish_exact1(root: Path, receipt: dict, stable_files: dict, stable_trees: dict,
                   stable_absences: dict, stable_pre: dict, fixture_hook=None) -> dict:
    global TERMINAL_COMMITTED
    prep = root.with_name(root.name + ".registration-prep")
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("authority root exists")
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    old_mask = None
    identity = None
    owned_receipt_identity = None
    owned_receipt_record = None
    try:
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        for sig in old_handlers:
            signal.signal(sig, signal_handler)
        os.mkdir(prep, 0o755)
        identity = (os.lstat(prep).st_dev, os.lstat(prep).st_ino)
        receipt_path = prep / "authority_receipt.json"
        write_exclusive(receipt_path, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
        receipt_metadata = os.lstat(receipt_path)
        owned_receipt_identity = (receipt_metadata.st_dev, receipt_metadata.st_ino)
        owned_receipt_record = regular(receipt_path)
        fsync_dir(prep)
        if tree(prep)["file_count"] != 1:
            raise RuntimeError("prep exact1")
        if snapshot(stable_files, stable_trees, stable_absences) != stable_pre:
            raise RuntimeError("prepromote drift")
        if fixture_hook is not None:
            fixture_hook("before_promote", prep, root)
        rename_directory_noreplace(prep, root)
        fsync_dir(root.parent)
        promoted = os.lstat(root)
        if (promoted.st_dev, promoted.st_ino) != identity or root.is_symlink() or not root.is_dir():
            raise RuntimeError("promoted authority ownership")
        final_tree = tree(root)
        if final_tree["file_count"] != 1 or final_tree["inventory"][0][0] != "authority_receipt.json":
            raise RuntimeError("final exact1")
        TERMINAL_COMMITTED = True
        return final_tree
    except BaseException:
        if identity is not None and os.path.lexists(prep):
            st = os.lstat(prep)
            if (st.st_dev, st.st_ino) != identity or prep.is_symlink() or not prep.is_dir():
                raise RuntimeError("authority prep ownership drift; refusing cleanup")
            members = list(prep.iterdir())
            if {path.name for path in members} - {"authority_receipt.json"}:
                raise RuntimeError("authority prep foreign member; refusing cleanup")
            for path in members:
                metadata = os.lstat(path)
                if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError("authority prep foreign type; refusing cleanup")
                if ((metadata.st_dev, metadata.st_ino) != owned_receipt_identity
                        or regular(path) != owned_receipt_record):
                    raise RuntimeError("authority prep member ownership drift; refusing cleanup")
                path.unlink()
            prep.rmdir()
            fsync_dir(prep.parent)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler if not TERMINAL_COMMITTED else signal.SIG_IGN)
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def publication_self_test() -> bool:
    global TERMINAL_COMMITTED
    if os.name != "posix":
        return True
    with tempfile.TemporaryDirectory(prefix="v503-authority-publish-") as raw:
        base = Path(raw)
        stable = snapshot({}, {}, {})
        root = base / "normal"
        TERMINAL_COMMITTED = False
        observed = publish_exact1(root, {"fixture": "normal"}, {}, {}, {}, stable)
        passed = observed["file_count"] == 1 and root.is_dir() and not root.is_symlink()

        raced = base / "foreign-root"
        TERMINAL_COMMITTED = False
        def create_foreign(_stage, _prep, final):
            final.mkdir()
        try:
            publish_exact1(raced, {"fixture": "race"}, {}, {}, {}, stable, create_foreign)
        except FileExistsError:
            passed = passed and raced.is_dir() and not raced.is_symlink() and not list(raced.iterdir())
            passed = passed and not os.path.lexists(raced.with_name(raced.name + ".registration-prep"))
        else:
            raise RuntimeError("foreign root race accepted")

        foreign = base / "foreign-prep"
        TERMINAL_COMMITTED = False
        def add_foreign(_stage, prep, _final):
            (prep / "foreign.bin").write_bytes(b"foreign")
            raise RuntimeError("fixture foreign prep")
        try:
            publish_exact1(foreign, {"fixture": "foreign"}, {}, {}, {}, stable, add_foreign)
        except RuntimeError as error:
            prep = foreign.with_name(foreign.name + ".registration-prep")
            passed = passed and "foreign member" in str(error) and (prep / "foreign.bin").read_bytes() == b"foreign"
            passed = passed and not os.path.lexists(foreign)
        else:
            raise RuntimeError("foreign prep cleanup accepted")

        foreign_type = base / "foreign-type"
        TERMINAL_COMMITTED = False
        def replace_with_foreign_type(_stage, prep, _final):
            receipt = prep / "authority_receipt.json"
            receipt.unlink()
            receipt.mkdir()
            raise RuntimeError("fixture foreign type")
        try:
            publish_exact1(foreign_type, {"fixture": "type"}, {}, {}, {}, stable, replace_with_foreign_type)
        except RuntimeError as error:
            prep = foreign_type.with_name(foreign_type.name + ".registration-prep")
            passed = passed and "foreign type" in str(error) and (prep / "authority_receipt.json").is_dir()
            passed = passed and not os.path.lexists(foreign_type)
        else:
            raise RuntimeError("foreign prep type cleanup accepted")

        replacement = base / "same-name-replacement"
        TERMINAL_COMMITTED = False
        def replace_with_same_name_regular(_stage, prep, _final):
            receipt = prep / "authority_receipt.json"
            receipt.unlink()
            receipt.write_bytes(b"foreign replacement")
            raise RuntimeError("fixture same-name replacement")
        try:
            publish_exact1(replacement, {"fixture": "replacement"}, {}, {}, {}, stable,
                           replace_with_same_name_regular)
        except RuntimeError as error:
            prep = replacement.with_name(replacement.name + ".registration-prep")
            receipt = prep / "authority_receipt.json"
            passed = passed and "member ownership drift" in str(error)
            passed = passed and receipt.read_bytes() == b"foreign replacement"
            passed = passed and not os.path.lexists(replacement)
        else:
            raise RuntimeError("same-name replacement cleanup accepted")
        TERMINAL_COMMITTED = False
        return passed


def build_context(args) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    if args.contract != CONTRACT_PATH or args.materializer_source != MATERIALIZER_PATH or args.authority_root != AUTHORITY_ROOT:
        raise RuntimeError("canonical CLI")
    contract_record = regular(args.contract, args.contract_sha, args.contract_bytes)
    materializer_record = regular(args.materializer_source, args.materializer_sha,
                                  args.materializer_bytes)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1645
            or pending(contract)):
        raise RuntimeError("contract schema")
    if contract["authority_materializer_source"] != materializer_record:
        raise RuntimeError("materializer binding")
    sources = {name: record(row) for name, row in contract["source_closure"].items()}
    if (set(sources) != set(CONTRACT_SOURCE_ORDER) or contract["source_role_order"] != CONTRACT_SOURCE_ORDER
            or contract["source_aliases"] != {k: SOURCE_ALIASES[k] for k in CONTRACT_SOURCE_ORDER}
            or contract["source_closure_sha256"] != csha(sources)
            or sources["authority_materializer"] != materializer_record):
        raise RuntimeError("source closure")
    fixed = {
        "reconciler_r2": (R2_PATH, "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377", 51845),
        "phase_a_design_contract": (PHASE_PATH, "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64", 43960),
        "repair_formal_receipt": (REPAIR_PATH, "b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b", 36181),
        "postregistration_static_receipt": (STATIC_PATH, "441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb", 49008),
    }
    for role, (path, digest, size) in fixed.items():
        if sources[role] != {"path": str(path), "sha256": digest, "logical_bytes": size}:
            raise RuntimeError("fixed source " + role)
    if Path(sources["direct_reconciler_wrapper"]["path"]) != WRAPPER_PATH:
        raise RuntimeError("wrapper path")
    if (contract["repair_formal_record"] != sources["repair_formal_receipt"]
            or contract["postregistration_static_record"] != sources["postregistration_static_receipt"]
            or contract["phase_a_design_contract_record"] != sources["phase_a_design_contract"]):
        raise RuntimeError("record aliases")

    f813 = verify_tree(contract["f813_registration_tree"])
    if (f813["root"] != str(F813_ROOT) or f813["file_count"] != 2
            or f813["logical_file_bytes"] != 27771
            or f813["sha256sum_lines_digest_sha256"] != "717d6ae12fbacbaafa147d03503140e5f807c028acc97a7294c84da2d79fe3fe"
            or f813["canonical_json_triples_digest_sha256"] != "a0602666a7e59d60e464b3bc76b31b2fbf501e3d37bbd0b18b2db8c1233971ec"):
        raise RuntimeError("F813 exact2")
    v502_receipt = record(contract["v502_authority_receipt"])
    v502_tree = verify_tree(contract["v502_authority_registration_tree"])
    v502_evidence = verify_tree(contract["v502_authority_materialization_evidence_tree"])
    v502_process_record = record(contract["v502_authority_materializer_process_receipt"])
    v502_process = json.loads(Path(v502_process_record["path"]).read_text())
    if ((v502_receipt["sha256"], v502_receipt["logical_bytes"]) !=
            ("83e238d5b0c579b7e95ad1cd377021ab5daeb95c2e00af8189768ef28cd4c3e7", 314856)
            or v502_tree["root"] != str(V502_AUTH_ROOT) or v502_tree["file_count"] != 1
            or v502_tree["sha256sum_lines_digest_sha256"] != "ab41d8214582dc0b69b94634f9658341aa23507138e4c4889a17e15a66528df7"
            or v502_tree["canonical_json_triples_digest_sha256"] != "cb406cc543854456ca9237f4e3daf7daeab4894437dbe9fdbaa52f4cabffa47e"
            or v502_evidence["root"] != str(V502_EVIDENCE_ROOT) or v502_evidence["file_count"] != 6
            or v502_evidence["logical_file_bytes"] != 203571
            or v502_evidence["sha256sum_lines_digest_sha256"] != "64045a24ece15e26976af33c79a71b6aca720bce47e3293d96fffe5b62cf9e4f"
            or v502_evidence["canonical_json_triples_digest_sha256"] != "f05e1f5e2490eb015926f2a2aabcc351e1102c5d46566e435658dc217e08a70f"
            or (v502_process_record["sha256"], v502_process_record["logical_bytes"]) !=
               ("f3c208ee3af1f09f0e34631cf56379137447b957c6077f73a7d93c0140eef54b", 103043)
            or v502_process.get("authority_materializer_invocations") != 1
            or any(v502_process.get(k) != 0 for k in
                   ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or v502_process.get("retry_authorized") is not False
            or v502_process.get("pre_post_snapshots_exactly_equal") is not True
            or v502_process.get("cleanup", {}).get("reaped") is not True
            or v502_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v502 authority closure")

    failure_record = record(contract["v502_failure_summary"])
    failure = json.loads(Path(failure_record["path"]).read_text())
    expected_ancestry = {
        "status": "frozen_reconstructed_v502_outer_preintent_v494_ancestry_failure_no_retry",
        "native_exit_code": 1, "outer_wrapper_invocations": 1,
        "corrected_inner_wrapper_invocations": 0, "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "root_cause": "the v502 outer embedded a stale two-leaf summary of the v494 materializer failure ancestry while the frozen v502 contract carried the later exact validator wording",
        "corrected_rule": "v503 compact authority binds the immutable failure JSON and does not replay legacy v494/v495 runtime validators",
    }
    predicates = failure.get("combined_predicate_evaluation", [])
    leafdiff = failure.get("json_leaf_diff", [])
    inv = failure.get("invocation_partition", {})
    if ((failure_record["path"], failure_record["sha256"], failure_record["logical_bytes"])
            != (str(FAILURE_PATH), "418f399030a0f0e13571c38a8e55ac36de45d2012616adef49d7452ab90a4ce9", 8521)
            or failure.get("status") != expected_ancestry["status"]
            or failure.get("native_exit_code") != 1 or failure.get("native_persistent_capture_available") is not False
            or len(predicates) != 4 or [row.get("passed") for row in predicates] != [True, True, True, False]
            or [row.get("path") for row in leafdiff] != [["corrected_rule"], ["root_cause"]]
            or inv.get("outer_wrapper_invocations") != 1
            or inv.get("corrected_inner_wrapper_invocations") != 0
            or inv.get("reconciler_r2_invocations") != 0
            or contract["v502_failure_ancestry"] != expected_ancestry
            or any(failure.get("authorization", {}).values())):
        raise RuntimeError("v502 failure summary")

    v504_failure_record = record(contract["v504_restore_failure_forensic"])
    v504_design_record = record(contract["v504_restore_design"])
    v504_helper_record = record(contract["v504_restore_helper"])
    v504_script_record = record(contract["v504_restore_script"])
    v504_evidence = verify_tree(contract["v504_restore_evidence_tree"])
    v504_process_record = record(contract["v504_restore_process_receipt"])
    v504_target_record = record(contract["v504_restored_target"])
    v504_failure = json.loads(Path(v504_failure_record["path"]).read_text())
    v504_process = json.loads(Path(v504_process_record["path"]).read_text())
    v504_ancestry = {
        "status": "passed_exact_once_volatile_alias_restored_no_reconciliation_execution",
        "failed_predecessor_helper_invocations": 1,
        "failed_predecessor_restore_invocations": 0,
        "successor_restore_invocations": 1,
        "direct_reconciler_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "pre_post_snapshots_exactly_equal": True,
    }
    if (
        v504_failure_record != {"path": str(V504_FAILURE_PATH), "sha256": "f04127d6e9b78860ddb873e3cb660e56c8f27141e0383bcfbf9aa6c7ee7709d8", "logical_bytes": 4772}
        or v504_design_record != {"path": str(V504_DESIGN_PATH), "sha256": "509d5badb6a7c0eefced7dd92f41fc16ffd8da63a8cce5884a473ec12af3155d", "logical_bytes": 5617}
        or v504_helper_record != {"path": str(V504_HELPER_PATH), "sha256": "9860843f8aa01e7665f02bddfa975efcf0d382cae4ad47f228fb630b2b046530", "logical_bytes": 38626}
        or v504_script_record != {"path": str(V504_SCRIPT_PATH), "sha256": "825fafea3f59f84b4f8d2d98cd790e2de8579de51d5f8ac36cded277baff57ca", "logical_bytes": 4077}
        or v504_evidence["root"] != str(V504_EVIDENCE_ROOT)
        or v504_evidence["file_count"] != 6
        or v504_evidence["logical_file_bytes"] != 76219
        or v504_evidence["sha256sum_lines_digest_sha256"] != "ae6e4ce22be7a7557699a019475005a07427bcc80d1135df4ef638e3b4a68642"
        or v504_evidence["canonical_json_triples_digest_sha256"] != "e23cbed8eacb50da40f843147b5aa9896e4b6b75a8d58d2b3b9fca34cf43e909"
        or v504_process_record != {"path": str(V504_EVIDENCE_ROOT / "process_receipt.json"), "sha256": "aaf5b944b93d94ca818712737b20289445d9ba6e9712bd0d42f9c38ab305b6d2", "logical_bytes": 22747}
        or v504_target_record != {"path": str(V504_TARGET_PATH), "sha256": "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b", "logical_bytes": 6475}
        or v504_failure.get("status") != "frozen_reconstructed_pre_evidence_failure_no_retry"
        or v504_failure.get("invocation_partition", {}).get("alias_restore_helper_invocations") != 1
        or v504_failure.get("invocation_partition", {}).get("alias_restore_invocations") != 0
        or v504_failure.get("authorization", {}).get("v503_alias_restore_retry_authorized") is not False
        or v504_process.get("status") != v504_ancestry["status"]
        or v504_process.get("passed") is not True
        or v504_process.get("restore_invocations") != 1
        or any(v504_process.get(k) != 0 for k in ("direct_reconciler_wrapper_invocations", "reconciler_r2_invocations", "phase_a_invocations", "training_invocations"))
        or v504_process.get("retry_authorized") is not False
        or v504_process.get("pre_post_snapshots_exactly_equal") is not True
        or v504_process.get("cleanup") != {"child_started": False, "reaped": True, "group_empty": True}
        or v504_process.get("target") != v504_target_record
        or v504_process.get("design") != v504_design_record
        or v504_process.get("transport_helper") != v504_helper_record
        or v504_process.get("transport_script") != v504_script_record
        or contract["v504_restore_ancestry"] != v504_ancestry
    ):
        raise RuntimeError("v504 restored alias closure")

    historical_keys = {
        "execution_authority_root", "execution_authority_prep", "wrapper_attempt_root",
        "wrapper_attempt_prep", "transparent_receipt", "transparent_receipt_tmp",
        "qualification_root", "v502_outer_root", "v502_outer_prep", "v502_inner_root",
        "v502_inner_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = decode_absences(contract["historical_absences"], historical_keys)
    current = decode_absences(contract["current_absences_after_authority"], current_keys)
    expected_paths = {
        "execution_authority_root": AUTHORITY_ROOT,
        "execution_authority_prep": AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep"),
        "wrapper_attempt_root": WRAPPER_ATTEMPT_ROOT,
        "wrapper_attempt_prep": WRAPPER_ATTEMPT_ROOT.with_name(WRAPPER_ATTEMPT_ROOT.name + ".attempt-prep"),
        "transparent_receipt": TRANSPARENT_PATH,
        "transparent_receipt_tmp": TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name + ".tmp"),
        "qualification_root": QUALIFICATION_ROOT,
        "v502_outer_root": J / "v502_v501_inner_binding_path_repaired_reconciliation_outer_execution_evidence_seed1644_20260825",
        "v502_outer_prep": J / "v502_v501_inner_binding_path_repaired_reconciliation_outer_execution_evidence_seed1644_20260825.outer-prep",
        "v502_inner_root": J / "v502_v501_inner_binding_path_repaired_reconciliation_inner_attempt_seed1644_20260825",
        "v502_inner_prep": J / "v502_v501_inner_binding_path_repaired_reconciliation_inner_attempt_seed1644_20260825.attempt-prep",
    }
    if historical != expected_paths or current != {k: v for k, v in expected_paths.items() if k in current_keys}:
        raise RuntimeError("absence paths")
    if any(os.path.lexists(path) for path in historical.values()):
        raise RuntimeError("absence prestate")
    fragments = {str(WRAPPER_PATH), str(R2_PATH), str(WRAPPER_ATTEMPT_ROOT)}
    if live_processes(fragments) or not gpu_empty():
        raise RuntimeError("process or GPU prestate")
    authority_sources = {"authority_design_contract": contract_record, **sources}
    files = {"contract": str(CONTRACT_PATH), "materializer": str(MATERIALIZER_PATH),
             "wrapper": str(WRAPPER_PATH), "r2": str(R2_PATH), "phase": str(PHASE_PATH),
             "repair": str(REPAIR_PATH), "static": str(STATIC_PATH), "failure": str(FAILURE_PATH),
             "v502_authority_receipt": v502_receipt["path"], "v502_process": v502_process_record["path"],
             "v504_failure": v504_failure_record["path"], "v504_design": v504_design_record["path"],
             "v504_helper": v504_helper_record["path"], "v504_script": v504_script_record["path"],
             "v504_process": v504_process_record["path"], "v504_target": v504_target_record["path"]}
    trees = {"f813": str(F813_ROOT), "v502_authority": str(V502_AUTH_ROOT),
             "v502_evidence": str(V502_EVIDENCE_ROOT), "v504_evidence": str(V504_EVIDENCE_ROOT)}
    return contract, contract_record, materializer_record, authority_sources, historical, current, {
        "files": files, "trees": trees, "f813": f813, "v502_receipt": v502_receipt,
        "v502_tree": v502_tree, "v502_evidence": v502_evidence,
        "v502_process": v502_process_record, "failure": failure_record,
        "failure_ancestry": expected_ancestry,
        "v504_failure": v504_failure_record, "v504_design": v504_design_record,
        "v504_helper": v504_helper_record, "v504_script": v504_script_record,
        "v504_evidence": v504_evidence, "v504_process": v504_process_record,
        "v504_target": v504_target_record, "v504_ancestry": v504_ancestry,
    }


def main() -> int:
    if os.sys.argv[1:] == ["--synthetic-self-test"]:
        checks = {
            "contract_top_count": len(CONTRACT_TOP_KEYS) == 33,
            "authority_top_count": len(AUTH_TOP_KEYS) == 46,
            "check_count": len(CHECK_KEYS) == 33,
            "contract_source_count": len(CONTRACT_SOURCE_ORDER) == 6,
            "authority_source_count": len(AUTHORITY_SOURCE_ORDER) == 7,
            "source_alias_bijection": set(SOURCE_ALIASES) == set(AUTHORITY_SOURCE_ORDER)
                                      and len(set(SOURCE_ALIASES.values())) == 7,
            "authorization_boundary": AUTHORIZATION["direct_single_reconciler_authorized"] is True
                                      and AUTHORIZATION["direct_r2_authorized"] is False
                                      and AUTHORIZATION["retry_authorized"] is False,
            "fresh_roots": "v503_" in AUTHORITY_ROOT.name and "v503_" in WRAPPER_ATTEMPT_ROOT.name,
            "publication_noreplace_owned_cleanup": publication_self_test(),
        }
        print(json.dumps({"passed": all(checks.values()), "checks": checks,
                          "checks_sha256": csha(checks)}, sort_keys=True))
        return 0 if all(checks.values()) else 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--contract-bytes", type=int, required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--materializer-bytes", type=int, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--read-only-preflight", action="store_true")
    args = parser.parse_args()
    contract, contract_record, materializer_record, authority_sources, historical, current, anchors = build_context(args)
    files, trees = anchors["files"], anchors["trees"]
    pre = snapshot(files, trees, current)
    post = snapshot(files, trees, current)
    if pre != post:
        raise RuntimeError("input drift")
    receipt = {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "source_closure": authority_sources, "source_role_order": AUTHORITY_SOURCE_ORDER,
        "source_aliases": SOURCE_ALIASES, "source_closure_sha256": csha(authority_sources),
        "authority_design_contract": contract_record, "authority_materializer_source": materializer_record,
        "direct_reconciler_wrapper_source": authority_sources["direct_reconciler_wrapper"],
        "reconciler_r2_source": authority_sources["reconciler_r2"],
        "phase_a_design_contract_source": authority_sources["phase_a_design_contract"],
        "repair_formal_source": authority_sources["repair_formal_receipt"],
        "postregistration_static_source": authority_sources["postregistration_static_receipt"],
        "repair_formal_record": contract["repair_formal_record"],
        "postregistration_static_record": contract["postregistration_static_record"],
        "phase_a_design_contract_record": contract["phase_a_design_contract_record"],
        "f813_registration_tree": anchors["f813"],
        "v502_authority_receipt": anchors["v502_receipt"],
        "v502_authority_registration_tree": anchors["v502_tree"],
        "v502_authority_materialization_evidence_tree": anchors["v502_evidence"],
        "v502_authority_materializer_process_receipt": anchors["v502_process"],
        "v502_failure_summary": anchors["failure"], "v502_failure_ancestry": anchors["failure_ancestry"],
        "v504_restore_failure_forensic": anchors["v504_failure"],
        "v504_restore_design": anchors["v504_design"],
        "v504_restore_helper": anchors["v504_helper"],
        "v504_restore_script": anchors["v504_script"],
        "v504_restore_evidence_tree": anchors["v504_evidence"],
        "v504_restore_process_receipt": anchors["v504_process"],
        "v504_restored_target": anchors["v504_target"],
        "v504_restore_ancestry": anchors["v504_ancestry"],
        "wrapper_attempt_root": str(WRAPPER_ATTEMPT_ROOT),
        "transparent_static_receipt_path": str(TRANSPARENT_PATH),
        "historical_absences": {k: {"path": str(v), "absent": True} for k, v in sorted(historical.items())},
        "required_absences": {k: {"path": str(v), "absent": True} for k, v in sorted(current.items())},
        "checks": {k: True for k in CHECK_KEYS}, "check_keys": CHECK_KEYS,
        "check_key_set_sha256": csha(CHECK_KEYS),
        "checks_sha256": csha({k: True for k in CHECK_KEYS}),
        "input_pre_snapshot": pre, "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True, "authorization": AUTHORIZATION,
        "runtime_observation": RUNTIME, "execution_boundary": EXECUTION_BOUNDARY,
    }
    schema = contract["authority_receipt_contract"]
    if (set(receipt) != AUTH_TOP_KEYS or set(schema) != {
        "format", "status", "top_keys", "check_keys", "check_key_set_sha256",
        "checks_sha256", "authorization_exact", "runtime_observation_exact"}
            or schema["format"] != OUTPUT_FORMAT or schema["status"] != OUTPUT_STATUS
            or schema["top_keys"] != sorted(AUTH_TOP_KEYS) or schema["check_keys"] != CHECK_KEYS
            or schema["check_key_set_sha256"] != csha(CHECK_KEYS)
            or schema["checks_sha256"] != csha({k: True for k in CHECK_KEYS})
            or schema["authorization_exact"] != AUTHORIZATION
            or schema["runtime_observation_exact"] != RUNTIME
            or contract["authorization"] != AUTHORIZATION
            or contract["runtime_observation"] != RUNTIME
            or contract["execution_boundary"] != EXECUTION_BOUNDARY):
        raise RuntimeError("receipt schema")
    if args.read_only_preflight:
        print(json.dumps({"status": "passed_read_only_preflight_no_materialization",
                          "contract_top_count": len(contract), "authority_top_count": len(receipt),
                          "check_count": len(CHECK_KEYS), "contract_source_count": len(contract["source_closure"]),
                          "authority_source_count": len(authority_sources), "input_snapshots_equal": pre == post},
                         sort_keys=True))
        return 0
    stable_absences = {k: v for k, v in current.items() if k != "execution_authority_prep"}
    stable_pre = snapshot(files, trees, stable_absences)
    publish_exact1(AUTHORITY_ROOT, receipt, files, trees, stable_absences, stable_pre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
