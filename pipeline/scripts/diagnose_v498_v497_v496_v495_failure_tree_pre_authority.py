#!/usr/bin/env python3
"""Register a read-only, pre-authority diagnosis of the v496 combined predicate failure."""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import signal
import stat
import symtable
import tempfile
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SELF_PATH = ROOT / "pipeline/scripts/diagnose_v498_v497_v496_v495_failure_tree_pre_authority.py"
DIAGNOSTIC_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1640_20260825"
DIAGNOSTIC_PREP = DIAGNOSTIC_ROOT.with_name(DIAGNOSTIC_ROOT.name + ".registration-prep")
DIAGNOSTIC_EVIDENCE_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1640_20260825"
DIAGNOSTIC_EVIDENCE_PREP = DIAGNOSTIC_EVIDENCE_ROOT.with_name(DIAGNOSTIC_EVIDENCE_ROOT.name + ".execution-prep")
OUTPUT_NAME = "diagnostic_receipt.json"

V495_FAILURE_ROOT = J / "v495_v494_v490_interpreter_validator_repair_execution_authority_materialization_evidence_seed1637_20260825"
V495_PROCESS_PATH = V495_FAILURE_ROOT / "process_receipt.json"
V495_EXPECTED_TREE = {
    "root": str(V495_FAILURE_ROOT),
    "inventory": [
        ["argv.json", "54cda5c2853e9b9d74c1abc11b3c56ef51de5a1650487d8813c4454fdc17b8a3", 16203],
        ["intent.json", "e339933407261a528b516d9bb3759fffdc853fc7998416b5f1e66b745a032828", 542],
        ["materializer_stderr.log", "bfe41948b189d59af0aa4644bc3be1d32d4a2dea8537d6b867ba3c51a4cdbc75", 544],
        ["materializer_stdout.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["process_receipt.json", "72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4", 411],
        ["transport_helper.py", "f2ad3a5a10fb419c0fdd1595997e212c5ec1ccfbb6f6ec3c23410d7ab9b31727", 14439],
    ],
    "file_count": 6,
    "logical_file_bytes": 32139,
    "sha256sum_lines_digest_sha256": "21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1b",
    "canonical_json_triples_digest_sha256": "9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290",
}
V495_PROCESS_RECORD = {
    "path": str(V495_PROCESS_PATH),
    "sha256": "72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4",
    "logical_bytes": 411,
}

V496_FAILURE_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_materialization_evidence_seed1638_20260825"
V496_EXPECTED_TREE = {
    "root": str(V496_FAILURE_ROOT),
    "inventory": [
        ["argv.json", "327cb32d890f937f7b0fec53902cf721af8df9a6195de9d82c01fbad7afb5f8b", 18856],
        ["intent.json", "c5bb3b6759f53a1de617f42cfbc343e6caf2dcb94916caba8cf3047bc025d06a", 557],
        ["materializer_stderr.log", "1ce5cb2055a08736e0a4927af4817f08bbd1c1e421e132574deff4bc70a471fa", 531],
        ["materializer_stdout.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["process_receipt.json", "5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d", 411],
        ["transport_helper.py", "df90feed74fa40eb0dba758cb285be9766fed0ca6ff8a79285a468ea40a23305", 14530],
    ],
    "file_count": 6,
    "logical_file_bytes": 34885,
    "sha256sum_lines_digest_sha256": "ab63f37f6275193a97db39cc71d37b32f4f653f87e78c375ddb460558b4e1f9e",
    "canonical_json_triples_digest_sha256": "c78038212aee67743f08dc936a539ad5d99a6ff931b87f12b5b294dfa4a0dac0",
}
V496_PROCESS_RECORD = {
    "path": str(V496_FAILURE_ROOT / "process_receipt.json"),
    "sha256": "5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d",
    "logical_bytes": 411,
}
V496_PROCESS_PATH = Path(V496_PROCESS_RECORD["path"])
V496_HELPER_RECORD = {
    "path": str(V496_FAILURE_ROOT / "transport_helper.py"),
    "sha256": "df90feed74fa40eb0dba758cb285be9766fed0ca6ff8a79285a468ea40a23305",
    "logical_bytes": 14530,
}
V496_SOURCE_RECORDS = {
    "authority_contract": {"path": str(ROOT / "pipeline/scripts/v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_contract.json"), "sha256": "33aad023e6275e21702c569261aec754e58c151622055ab3beb0c81434ac335c", "logical_bytes": 46664},
    "authority_materializer": {"path": str(ROOT / "pipeline/scripts/materialize_v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority.py"), "sha256": "6a38efcb9e714cf58a694bac62eadd86cabc6e746d0fdd1195d168963de1cf06", "logical_bytes": 62492},
    "inner_wrapper": {"path": str(ROOT / "pipeline/scripts/launch_v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner.py"), "sha256": "c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f", "logical_bytes": 101600},
    "outer_wrapper": {"path": str(ROOT / "pipeline/scripts/launch_v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer.py"), "sha256": "09d5b725ba45bcf274d134fd40fdbea52795e68f6a8193149085b96d563a07c2", "logical_bytes": 61068},
}

FAILED_V497_SOURCE_RECORD = {"path": str(ROOT / "pipeline/scripts/diagnose_v497_v496_v495_failure_tree_pre_authority.py"), "sha256": "6ad8f54b53fc2178330f27e9bc772de8fc7357a27752f17a309b835f8f796ecb", "logical_bytes": 26978}
FAILED_V497_HELPER_RECORD = {"path": str(ROOT / "pipeline/scripts/invoke_v497_v496_v495_failure_tree_pre_authority_diagnostic_once.py"), "sha256": "a8da83f408fa30c241b311caf86c4d0d7b065099dc7445dea0a1d75e192b44ed", "logical_bytes": 27893}
FAILED_V497_SCRIPT_RECORD = {"path": "/root/v497_pre_authority_diagnostic_once.sh", "sha256": "eb0f0908c5c2185627bfb59fa02b46279ba929deb04d06253e0d36835c6f4104", "logical_bytes": 1166}
FAILED_V497_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1639_20260825"
FAILED_V497_PREP = FAILED_V497_ROOT.with_name(FAILED_V497_ROOT.name + ".registration-prep")
FAILED_V497_EVIDENCE_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1639_20260825"
FAILED_V497_EVIDENCE_TREE = {
    "root": str(FAILED_V497_EVIDENCE_ROOT),
    "inventory": [
        ["argv.json", "7a2ce339eaf422310a5f42046bd3fb66983c74e3f582f3032e5e5a2f0d1ad315", 11350],
        ["diagnostic_stderr.log", "43077e085d04192f62354c584a19fdff5a5ea0c34e0cd160cd226d2b1ff69d27", 560],
        ["diagnostic_stdout.log", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0],
        ["intent.json", "79927b48f27cdc5d48454712863fe798e3baef4ab3efea1ac3d978cfaa399e83", 540],
        ["process_receipt.json", "5ad649b013fe0e9b17af76992950c53f8043a536ae3f2ecb04f6c9f65f443425", 459],
        ["transport_helper.py", "a8da83f408fa30c241b311caf86c4d0d7b065099dc7445dea0a1d75e192b44ed", 27893],
    ],
    "file_count": 6, "logical_file_bytes": 40802,
    "sha256sum_lines_digest_sha256": "ed2c81a06f71937581b451f84e137640310b363f04d569fe1bf0da1a60377b8d",
    "canonical_json_triples_digest_sha256": "02337d3883ddb46b16bfbf9d1e157dcc16ff4e462810e0de2e22a25082e5913d",
}
FAILED_V497_PROCESS_RECORD = {"path": str(FAILED_V497_EVIDENCE_ROOT / "process_receipt.json"), "sha256": "5ad649b013fe0e9b17af76992950c53f8043a536ae3f2ecb04f6c9f65f443425", "logical_bytes": 459}
FAILED_V497_STDERR_RECORD = {"path": str(FAILED_V497_EVIDENCE_ROOT / "diagnostic_stderr.log"), "sha256": "43077e085d04192f62354c584a19fdff5a5ea0c34e0cd160cd226d2b1ff69d27", "logical_bytes": 560}

V496_AUTHORITY_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_seed1638_20260825"
V496_OUTER_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer_execution_evidence_seed1638_20260825"
V496_INNER_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner_attempt_seed1638_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

PREDICATE_NAMES = [
    "root_exact", "file_count_exact", "logical_file_bytes_exact",
    "sha256sum_lines_digest_exact", "canonical_json_triples_digest_exact",
    "process_receipt_inventory_row_exact", "process_receipt_sha256_exact",
    "process_receipt_logical_bytes_exact",
]
FORMAT = "strict-track2-v498-v497-v496-v495-failure-tree-pre-authority-diagnostic-v1"
PASS_STATUS = "passed_no_execution_authority"
FAIL_STATUS = "failed_no_retry"
AUTHORIZATION = {
    "diagnostic_authorized": False,
    "authority_materialization_authorized": False,
    "outer_execution_authorized": False,
    "corrected_inner_wrapper_authorized": False,
    "reconciler_r2_authorized": False,
    "retry_authorized": False,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
    "submission_authorized": False,
}
RUNTIME = {
    "diagnostic_invocations": 1,
    "authority_materializer_invocations": 0,
    "outer_wrapper_invocations": 0,
    "corrected_inner_wrapper_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
RECEIPT_TOP_KEYS = {
    "format", "status", "passed", "seed", "diagnostic_source", "v496_authority_sources",
    "v496_failure_tree", "v496_failure_process_receipt", "v496_failure_transport_helper",
    "v496_failure_no_retry", "failed_v497_ancestry", "root_cause_boundary", "predicate_names",
    "predicate_name_key_set_sha256", "predicates", "predicates_sha256", "all_predicates_passed",
    "snapshot_1", "snapshot_2", "snapshots_exactly_equal", "required_absences",
    "runtime_observation", "authorization",
}
ABSENCE_PATHS = {
    "v496_authority_root": V496_AUTHORITY_ROOT,
    "v496_authority_prep": V496_AUTHORITY_ROOT.with_name(V496_AUTHORITY_ROOT.name + ".registration-prep"),
    "v496_outer_root": V496_OUTER_ROOT,
    "v496_outer_prep": V496_OUTER_ROOT.with_name(V496_OUTER_ROOT.name + ".outer-prep"),
    "v496_inner_root": V496_INNER_ROOT,
    "v496_inner_prep": V496_INNER_ROOT.with_name(V496_INNER_ROOT.name + ".attempt-prep"),
    "transparent_output": TRANSPARENT_PATH,
    "transparent_tmp": TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name + ".tmp"),
    "qualification_root": QUALIFICATION_ROOT,
    "failed_v497_diagnostic_root": FAILED_V497_ROOT,
    "failed_v497_diagnostic_prep": FAILED_V497_PREP,
}


class ControlledSignal(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"controlled signal {signum}")
        self.signum = signum


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def csha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path | str, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict:
    path = Path(path)
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"not canonical regular file: {path}")
    observed = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if expected_sha is not None and observed["sha256"] != expected_sha:
        raise RuntimeError(f"sha mismatch: {path}")
    if expected_bytes is not None and observed["logical_bytes"] != expected_bytes:
        raise RuntimeError(f"byte mismatch: {path}")
    return observed


def enumerate_tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            mode = os.lstat(path).st_mode
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"tree nonregular: {path}")
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree unsupported node: {path}")
    lines = "".join(f"{digest}  {rel}\n" for rel, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "root": str(root),
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def build_failure_snapshot(root: Path, process_path: Path) -> dict:
    """Build a fresh object from a fresh full enumeration and a separate regular-file read."""
    return {**enumerate_tree(root), "process_receipt_record": regular(process_path)}


def value_type(value: object) -> str:
    return type(value).__name__


def named_failure_tree_predicates(snapshot: dict, expected_tree: dict, expected_process_record: dict) -> list[dict]:
    """Pure, ordered predicate evaluator shared with the future authority materializer."""
    process_row = ["process_receipt.json", expected_process_record["sha256"], expected_process_record["logical_bytes"]]
    observed_process_row = next((row for row in snapshot["inventory"] if row[0] == "process_receipt.json"), None)
    pairs = [
        ("root_exact", expected_tree["root"], snapshot["root"]),
        ("file_count_exact", expected_tree["file_count"], snapshot["file_count"]),
        ("logical_file_bytes_exact", expected_tree["logical_file_bytes"], snapshot["logical_file_bytes"]),
        ("sha256sum_lines_digest_exact", expected_tree["sha256sum_lines_digest_sha256"], snapshot["sha256sum_lines_digest_sha256"]),
        ("canonical_json_triples_digest_exact", expected_tree["canonical_json_triples_digest_sha256"], snapshot["canonical_json_triples_digest_sha256"]),
        ("process_receipt_inventory_row_exact", process_row, observed_process_row),
        ("process_receipt_sha256_exact", expected_process_record["sha256"], snapshot["process_receipt_record"]["sha256"]),
        ("process_receipt_logical_bytes_exact", expected_process_record["logical_bytes"], snapshot["process_receipt_record"]["logical_bytes"]),
    ]
    if [name for name, _, _ in pairs] != PREDICATE_NAMES:
        raise RuntimeError("predicate order")
    return [
        {"name": name, "expected": expected, "observed": observed,
         "expected_type": value_type(expected), "observed_type": value_type(observed),
         "passed": type(expected) is type(observed) and expected == observed}
        for name, expected, observed in pairs
    ]


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if os.path.lexists(temporary):
        raise RuntimeError(f"temporary exists: {temporary}")
    data = canonical(value) + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
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


def publish_exact1(root: Path, prep: Path, receipt: dict, immutable_snapshot, phase_hook=None) -> dict:
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("diagnostic registration prestate")
    phase_hook = phase_hook or (lambda _phase: None)
    immutable_pre = immutable_snapshot()
    prep_created = False
    prep_identity = None
    postcommit_verified = False
    received_signal = None

    def on_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if not postcommit_verified:
            raise ControlledSignal(signum)

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, on_signal)
    baseline_mask = None
    signals_blocked = False
    if hasattr(signal, "pthread_sigmask"):
        baseline_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        signals_blocked = True
    try:
        prep.mkdir()
        prep_identity = directory_identity(prep)
        prep_created = True
        fsync_dir(prep.parent)
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
            signals_blocked = False
        phase_hook("owned_prep_unblocked")
        atomic_json(prep / OUTPUT_NAME, receipt)
        fsync_dir(prep)
        phase_hook("after_write")
        if immutable_snapshot() != immutable_pre:
            raise RuntimeError("post-write input drift")
        if directory_identity(prep) != prep_identity or os.path.lexists(root):
            raise RuntimeError("diagnostic ownership drift")
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            signals_blocked = True
        phase_hook("commit_window_blocked")
        os.replace(prep, root)
        fsync_dir(root.parent)
        prep_created = False
        phase_hook("after_promote_before_snapshot")
        immutable_after = immutable_snapshot()
        root_tree = enumerate_tree(root)
        output = root / OUTPUT_NAME
        if (immutable_after != immutable_pre or root_tree["inventory"] != [[OUTPUT_NAME, sha(output), output.stat().st_size]]
                or root_tree["file_count"] != 1 or os.path.lexists(prep)):
            raise RuntimeError("post-promote registration/input drift")
        postcommit_verified = True
    except BaseException:
        if prep_created and prep_identity is not None and not os.path.lexists(root):
            cleanup_owned_prep(prep, prep_identity)
        raise
    finally:
        if signals_blocked and baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return {"committed_success": True, "deferred_signal": received_signal,
            "registration_tree": root_tree, "immutable_input_snapshot": immutable_pre}


def required_absences() -> dict:
    return {name: {"path": str(path), "absent": not os.path.lexists(path)} for name, path in ABSENCE_PATHS.items()}


def failed_v497_ancestry() -> dict:
    tree = enumerate_tree(FAILED_V497_EVIDENCE_ROOT)
    if tree != FAILED_V497_EVIDENCE_TREE:
        raise RuntimeError("failed v497 execution exact6")
    process_record = regular(FAILED_V497_PROCESS_RECORD["path"], FAILED_V497_PROCESS_RECORD["sha256"], FAILED_V497_PROCESS_RECORD["logical_bytes"])
    stderr_record = regular(FAILED_V497_STDERR_RECORD["path"], FAILED_V497_STDERR_RECORD["sha256"], FAILED_V497_STDERR_RECORD["logical_bytes"])
    process = json.loads(Path(process_record["path"]).read_text())
    if (set(process) != {"authority_materializer_invocations", "cleanup", "corrected_inner_wrapper_invocations",
                         "diagnostic_invocations", "error", "error_type", "format", "outer_wrapper_invocations",
                         "passed", "reconciler_r2_invocations", "retry_authorized", "status"}
            or process["status"] != "failed_no_retry" or process["passed"] is not False
            or type(process["diagnostic_invocations"]) is not int or process["diagnostic_invocations"] != 1
            or type(process["authority_materializer_invocations"]) is not int or process["authority_materializer_invocations"] != 0
            or type(process["outer_wrapper_invocations"]) is not int or process["outer_wrapper_invocations"] != 0
            or type(process["corrected_inner_wrapper_invocations"]) is not int or process["corrected_inner_wrapper_invocations"] != 0
            or type(process["reconciler_r2_invocations"]) is not int or process["reconciler_r2_invocations"] != 0
            or process["retry_authorized"] is not False or process["cleanup"].get("reaped") is not True
            or process["cleanup"].get("group_empty") is not True
            or process.get("error") != "diagnostic rc 1"):
        raise RuntimeError("failed v497 process semantics")
    stderr_text = Path(stderr_record["path"]).read_text()
    if "NameError: name 'V496_PROCESS_PATH' is not defined" not in stderr_text:
        raise RuntimeError("failed v497 root cause")
    return {
        "diagnostic_source": regular(FAILED_V497_SOURCE_RECORD["path"], FAILED_V497_SOURCE_RECORD["sha256"], FAILED_V497_SOURCE_RECORD["logical_bytes"]),
        "transport_helper": regular(FAILED_V497_HELPER_RECORD["path"], FAILED_V497_HELPER_RECORD["sha256"], FAILED_V497_HELPER_RECORD["logical_bytes"]),
        "transport_script": regular(FAILED_V497_SCRIPT_RECORD["path"], FAILED_V497_SCRIPT_RECORD["sha256"], FAILED_V497_SCRIPT_RECORD["logical_bytes"]),
        "execution_tree": tree, "process_receipt": process_record, "stderr": stderr_record,
        "no_retry": True, "diagnostic_root_absent": not os.path.lexists(FAILED_V497_ROOT),
        "diagnostic_prep_absent": not os.path.lexists(FAILED_V497_PREP),
    }


def immutable_input_snapshot(self_record: dict) -> dict:
    sources = {name: regular(row["path"], row["sha256"], row["logical_bytes"])
               for name, row in V496_SOURCE_RECORDS.items()}
    return {
        "diagnostic_source": regular(self_record["path"], self_record["sha256"], self_record["logical_bytes"]),
        "v495_failure_snapshot": build_failure_snapshot(V495_FAILURE_ROOT, V495_PROCESS_PATH),
        "v496_failure_tree": enumerate_tree(V496_FAILURE_ROOT),
        "v496_failure_process_receipt": regular(V496_PROCESS_RECORD["path"], V496_PROCESS_RECORD["sha256"], V496_PROCESS_RECORD["logical_bytes"]),
        "v496_failure_transport_helper": regular(V496_HELPER_RECORD["path"], V496_HELPER_RECORD["sha256"], V496_HELPER_RECORD["logical_bytes"]),
        "v496_sources": sources,
        "failed_v497_ancestry": failed_v497_ancestry(),
        "required_absences": required_absences(),
    }


def predicate_tamper_fixture(snapshot: dict) -> dict:
    checks = {}
    mutations = {
        "root": {**snapshot, "root": snapshot["root"] + ".tamper"},
        "file_count": {**snapshot, "file_count": snapshot["file_count"] + 1},
        "logical_file_bytes": {**snapshot, "logical_file_bytes": snapshot["logical_file_bytes"] + 1},
        "sha256sum_lines_digest": {**snapshot, "sha256sum_lines_digest_sha256": "0" * 64},
        "canonical_json_triples_digest": {**snapshot, "canonical_json_triples_digest_sha256": "0" * 64},
        "process_receipt_inventory_row": {**snapshot, "inventory": [row if row[0] != "process_receipt.json" else [row[0], "0" * 64, row[2]] for row in snapshot["inventory"]]},
        "process_receipt_sha256": {**snapshot, "process_receipt_record": {**snapshot["process_receipt_record"], "sha256": "0" * 64}},
        "process_receipt_logical_bytes": {**snapshot, "process_receipt_record": {**snapshot["process_receipt_record"], "logical_bytes": snapshot["process_receipt_record"]["logical_bytes"] + 1}},
    }
    baseline = named_failure_tree_predicates(snapshot, V495_EXPECTED_TREE, V495_PROCESS_RECORD)
    checks["live_baseline_all_pass"] = all(row["passed"] for row in baseline)
    checks["live_baseline_order_exact"] = [row["name"] for row in baseline] == PREDICATE_NAMES
    for name, mutated in mutations.items():
        result = named_failure_tree_predicates(mutated, V495_EXPECTED_TREE, V495_PROCESS_RECORD)
        checks[name + "_tamper_rejected"] = sum(not row["passed"] for row in result) >= 1
    boolean_tamper = {**snapshot, "file_count": True}
    checks["bool_type_tamper_rejected"] = not named_failure_tree_predicates(boolean_tamper, V495_EXPECTED_TREE, V495_PROCESS_RECORD)[1]["passed"]
    return {"passed": all(checks.values()), "checks": checks, "check_count": len(checks), "evidence_sha256": csha(checks)}


def derived_process_path_ast_gate(source_text: str) -> bool:
    tree = ast.parse(source_text)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "V496_PROCESS_PATH"]
    if len(assignments) != 1:
        raise RuntimeError("V496_PROCESS_PATH assignment count")
    value = assignments[0].value
    if (not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "Path"
            or value.keywords or len(value.args) != 1):
        raise RuntimeError("V496_PROCESS_PATH Path call")
    argument = value.args[0]
    if (not isinstance(argument, ast.Subscript) or not isinstance(argument.value, ast.Name)
            or argument.value.id != "V496_PROCESS_RECORD"
            or not isinstance(argument.slice, ast.Constant) or argument.slice.value != "path"):
        raise RuntimeError("V496_PROCESS_PATH derivation")
    return True


def uppercase_global_resolution_gate(source_text: str) -> bool:
    tree = ast.parse(source_text)
    module_definitions = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_definitions.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_definitions.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    module_definitions.add(target.id)
    builtin_names = set(dir(builtins))
    table = symtable.symtable(source_text, "diagnostic.py", "exec")
    tables = [table]
    while tables:
        current = tables.pop()
        tables.extend(current.get_children())
        for symbol in current.get_symbols():
            name = symbol.get_name()
            if (name.isupper() and symbol.is_referenced() and symbol.is_global()
                    and name not in module_definitions and name not in builtin_names):
                raise RuntimeError(f"undefined uppercase global: {name}")
    derived_process_path_ast_gate(source_text)
    return True


def build_receipt_and_prestate(self_record: dict, publish_hook=None) -> tuple[dict, dict, bool]:
    """The sole production prepublish builder, also called by the live read-only fixture."""
    publish_hook = publish_hook or (lambda _receipt, _snapshot: None)
    if os.path.lexists(DIAGNOSTIC_ROOT) or os.path.lexists(DIAGNOSTIC_PREP):
        raise RuntimeError("fresh diagnostic prestate")
    snapshot_1 = build_failure_snapshot(V495_FAILURE_ROOT, V495_PROCESS_PATH)
    snapshot_2 = build_failure_snapshot(V495_FAILURE_ROOT, V495_PROCESS_PATH)
    predicates = named_failure_tree_predicates(snapshot_2, V495_EXPECTED_TREE, V495_PROCESS_RECORD)
    snapshots_equal = snapshot_1 == snapshot_2
    all_pass = snapshots_equal and all(row["passed"] for row in predicates)
    immutable_pre = immutable_input_snapshot(self_record)
    if immutable_pre["v495_failure_snapshot"] != snapshot_2:
        raise RuntimeError("third independent v495 snapshot drift")
    if immutable_pre["v496_failure_tree"] != V496_EXPECTED_TREE:
        raise RuntimeError("v496 failure exact6")
    if not all(row["absent"] for row in immutable_pre["required_absences"].values()):
        raise RuntimeError("required absence")
    process_record = regular(V496_PROCESS_PATH, V496_PROCESS_RECORD["sha256"], V496_PROCESS_RECORD["logical_bytes"])
    if process_record != immutable_pre["v496_failure_process_receipt"]:
        raise RuntimeError("v496 process record")
    process = json.loads(V496_PROCESS_PATH.read_text())
    if (process.get("status") != "failed_no_retry" or process.get("passed") is not False
            or type(process.get("materializer_invocations")) is not int or process["materializer_invocations"] != 1
            or type(process.get("outer_wrapper_invocations")) is not int or process["outer_wrapper_invocations"] != 0
            or type(process.get("corrected_inner_wrapper_invocations")) is not int or process["corrected_inner_wrapper_invocations"] != 0
            or type(process.get("r2_invocations")) is not int or process["r2_invocations"] != 0
            or process.get("retry_authorized") is not False
            or process.get("cleanup", {}).get("reaped") is not True
            or process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v496 failure semantics")
    receipt = {
        "format": FORMAT, "status": PASS_STATUS if all_pass else FAIL_STATUS, "passed": all_pass,
        "seed": 1640, "diagnostic_source": self_record,
        "v496_authority_sources": immutable_pre["v496_sources"],
        "v496_failure_tree": immutable_pre["v496_failure_tree"],
        "v496_failure_process_receipt": immutable_pre["v496_failure_process_receipt"],
        "v496_failure_transport_helper": immutable_pre["v496_failure_transport_helper"],
        "v496_failure_no_retry": True, "failed_v497_ancestry": immutable_pre["failed_v497_ancestry"],
        "root_cause_boundary": "v496 combined predicate failed; specific transient predicate unavailable because v496 did not persist per-predicate diagnostics; read-only postmortem all eight pass",
        "predicate_names": PREDICATE_NAMES, "predicate_name_key_set_sha256": csha(sorted(PREDICATE_NAMES)),
        "predicates": predicates, "predicates_sha256": csha(predicates),
        "all_predicates_passed": all(row["passed"] for row in predicates),
        "snapshot_1": snapshot_1, "snapshot_2": snapshot_2,
        "snapshots_exactly_equal": snapshots_equal, "required_absences": immutable_pre["required_absences"],
        "runtime_observation": RUNTIME, "authorization": AUTHORIZATION,
    }
    if set(receipt) != RECEIPT_TOP_KEYS:
        raise RuntimeError("receipt schema")
    publish_hook(receipt, immutable_pre)
    return receipt, immutable_pre, all_pass


def publish_fixture() -> dict:
    checks = {}
    with tempfile.TemporaryDirectory(prefix="v497-diagnostic-publish-") as directory:
        base = Path(directory)

        def case(name):
            parent = base / name
            parent.mkdir()
            guard = parent / "guard"
            guard.write_bytes(b"immutable\n")
            with guard.open("rb") as stream:
                os.fsync(stream.fileno())
            fsync_dir(parent)
            return parent / "root", parent / "root.registration-prep", guard, lambda: regular(guard)

        root, prep, _, snapshot = case("normal")
        result = publish_exact1(root, prep, {"passed": True}, snapshot)
        checks["normal_commit"] = result["committed_success"] and result["registration_tree"]["file_count"] == 1 and not os.path.lexists(prep)

        root, prep, _, snapshot = case("signal")
        def signal_hook(phase):
            if phase == "owned_prep_unblocked":
                if hasattr(signal, "pthread_sigmask"):
                    os.kill(os.getpid(), signal.SIGTERM)
                else:
                    raise ControlledSignal(signal.SIGTERM)
        try:
            publish_exact1(root, prep, {"passed": True}, snapshot, signal_hook)
            checks["precommit_signal_cleanup"] = False
        except ControlledSignal:
            checks["precommit_signal_cleanup"] = not os.path.lexists(root) and not os.path.lexists(prep)

        root, prep, _, snapshot = case("deferred")
        if hasattr(signal, "pthread_sigmask"):
            result = publish_exact1(root, prep, {"passed": True}, snapshot,
                                    lambda phase: os.kill(os.getpid(), signal.SIGTERM) if phase == "commit_window_blocked" else None)
            checks["commit_signal_deferred_success"] = result["committed_success"] and result["deferred_signal"] == signal.SIGTERM
        else:
            checks["commit_signal_deferred_success"] = os.name == "nt"

        root, prep, guard, snapshot = case("drift")
        def drift_hook(phase):
            if phase == "after_promote_before_snapshot":
                with guard.open("ab") as stream:
                    stream.write(b"drift\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        try:
            publish_exact1(root, prep, {"passed": True}, snapshot, drift_hook)
            checks["postpromote_drift_rejected"] = False
        except RuntimeError:
            checks["postpromote_drift_rejected"] = os.path.lexists(root) and not os.path.lexists(prep)
    return {"passed": all(checks.values()), "checks": checks, "check_count": len(checks), "evidence_sha256": csha(checks)}


def synthetic_self_test() -> int:
    source_path = Path(__file__).resolve()
    self_record = regular(source_path)
    first = build_failure_snapshot(V495_FAILURE_ROOT, V495_PROCESS_PATH)
    second = build_failure_snapshot(V495_FAILURE_ROOT, V495_PROCESS_PATH)
    predicate_fixture = predicate_tamper_fixture(first)
    publish = publish_fixture()
    hook_calls = []
    def stop_before_publish(receipt, snapshot):
        hook_calls.append({"receipt": receipt, "snapshot": snapshot})
    receipt, _, all_pass = build_receipt_and_prestate(self_record, stop_before_publish)
    source_text = source_path.read_text()
    global_gate_pass = uppercase_global_resolution_gate(source_text)
    without_definition = "\n".join(line for line in source_text.splitlines()
                                    if not line.startswith("V496_PROCESS_PATH = Path(V496_PROCESS_RECORD")) + "\n"
    typo = source_text.replace("json.loads(V496_PROCESS_PATH.read_text())", "json.loads(V496_PROCESS_PATX.read_text())", 1)
    try:
        uppercase_global_resolution_gate(without_definition)
        missing_definition_rejected = False
    except RuntimeError:
        missing_definition_rejected = True
    try:
        uppercase_global_resolution_gate(typo)
        function_typo_rejected = False
    except RuntimeError:
        function_typo_rejected = True
    checks = {
        "snapshots_equal": first == second,
        "snapshots_distinct_objects": first is not second and first["inventory"] is not second["inventory"] and first["process_receipt_record"] is not second["process_receipt_record"],
        "predicate_fixture": predicate_fixture["passed"],
        "publish_fixture": publish["passed"],
        "signal_symbols_live": signal.getsignal(signal.SIGINT) is not None and signal.getsignal(signal.SIGTERM) is not None,
        "production_build_receipt_pass": all_pass and receipt["passed"] is True and set(receipt) == RECEIPT_TOP_KEYS,
        "publish_hook_exact_once": len(hook_calls) == 1,
        "publish_hook_zero_state": (not os.path.lexists(DIAGNOSTIC_ROOT) and not os.path.lexists(DIAGNOSTIC_PREP)
                                    and not os.path.lexists(DIAGNOSTIC_EVIDENCE_ROOT) and not os.path.lexists(DIAGNOSTIC_EVIDENCE_PREP)),
        "uppercase_global_gate_pass": global_gate_pass,
        "missing_process_path_definition_rejected": missing_definition_rejected,
        "function_scope_typo_rejected": function_typo_rejected,
    }
    result = {"passed": all(checks.values()), "checks": checks,
              "predicate_fixture": predicate_fixture, "publish_fixture": publish,
              "production_receipt_sha256": csha(receipt),
              "evidence_sha256": csha({"checks": checks, "predicate_fixture": predicate_fixture, "publish_fixture": publish})}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


def main() -> int:
    if os.sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-source", type=Path, required=True)
    parser.add_argument("--diagnostic-source-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    self_path = Path(__file__)
    if (args.diagnostic_source != SELF_PATH or args.diagnostic_source.resolve() != SELF_PATH
            or self_path.resolve() != SELF_PATH or args.diagnostic_source_sha != sha(SELF_PATH)):
        raise RuntimeError("diagnostic self binding")
    if args.output_root != DIAGNOSTIC_ROOT or args.output_root.resolve() != DIAGNOSTIC_ROOT:
        raise RuntimeError("diagnostic output binding")
    if os.path.lexists(DIAGNOSTIC_ROOT) or os.path.lexists(DIAGNOSTIC_PREP):
        raise RuntimeError("diagnostic registration already consumed")

    self_record = regular(SELF_PATH, args.diagnostic_source_sha, SELF_PATH.stat().st_size)
    receipt, _, all_pass = build_receipt_and_prestate(self_record)
    publish = publish_exact1(DIAGNOSTIC_ROOT, DIAGNOSTIC_PREP, receipt,
                             lambda: immutable_input_snapshot(self_record))
    print(json.dumps({"status": receipt["status"], "passed": receipt["passed"],
                      "registration_tree": publish["registration_tree"],
                      "deferred_signal": publish["deferred_signal"]}, sort_keys=True), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
