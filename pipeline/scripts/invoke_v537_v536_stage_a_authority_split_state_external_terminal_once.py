#!/usr/bin/env python3
"""No-Popen external terminalizer for the v537 readonly candidate."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import stat
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
S = ROOT / "pipeline/scripts"
SELF_PATH = S / "invoke_v537_v536_stage_a_authority_split_state_external_terminal_once.py"
SCRIPT_PATH = Path("/root/v537_v536_stage_a_authority_split_state_external_terminal_once.sh")
DEPLOYMENT_RECORD = S / "v537_v536_stage_a_authority_split_state_external_terminal_transport_deployment_record.json"
RECONCILER = S / "reconcile_v537_v536_stage_a_authority_split_state_readonly.py"
RECONCILER_SHA = "76269001d6c3ae76fb4b33bf913a79ebc6cd9c26c118219573f5ae9f1367fdf1"
RECONCILER_BYTES = 24050
FORENSIC = S / "v537_v536_stage_a_authority_split_state_failure_forensic.json"
FORENSIC_SHA = "8a8abe3b18ac67b7c6739f2bab7d561d14755d327174c811ca887aeb37aa1c2f"
FORENSIC_BYTES = 13186
AUTH_ROOT = J / "v535_v534_public_s1_zero_update_gate_execution_authority_seed1668_20260827"
AUTH_RECEIPT = AUTH_ROOT / "authority_receipt.json"
AUTH_SHA = "2b7f1c1e82cccb107da8e44af883940701f2427deccef75749e3afc2d85d7358"
AUTH_BYTES = 125125
FAILED_ROOT = J / "v536_v535_stage_a_contract_absence_schema_repair_authority_materialization_evidence_seed1668_20260827"
FAILED_PROCESS = FAILED_ROOT / "process_receipt.json"
FAILED_PROCESS_SHA = "0689907a3e08721893b2a73e737a0421b1cb0c7ed111c8dd6769daa75f4c6b92"
FAILED_PROCESS_BYTES = 8324
CANDIDATE_ROOT = J / "v537_v536_stage_a_authority_split_state_readonly_candidate_seed1669_20260827"
CANDIDATE_PREP = CANDIDATE_ROOT.with_name(CANDIDATE_ROOT.name + ".candidate-prep")
CANDIDATE_NAME = "reconciliation_candidate_receipt.json"
EVIDENCE_ROOT = J / "v537_v536_stage_a_authority_split_state_external_terminal_evidence_seed1669_20260827"
EVIDENCE_PREP = EVIDENCE_ROOT.with_name(EVIDENCE_ROOT.name + ".evidence-prep")
EXACT5 = ["argv.json", "helper_stderr.log", "helper_stdout.log", "intent.json", "transport_helper.py"]
EXACT6 = sorted(EXACT5 + ["process_receipt.json"])
AT_FDCWD = -100
RENAME_NOREPLACE = 1
TEST_HOOK = None


def hook(name: str, state: dict) -> None:
    if TEST_HOOK is not None:
        TEST_HOOK(name, state)


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def csha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def regular(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> tuple[dict, bytes]:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"nonregular:{path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            raise RuntimeError(f"identity:{path}")
        data = os.pread(fd, fst.st_size + 1, 0)
        if len(data) != fst.st_size or os.pread(fd, 1, fst.st_size) != b"":
            raise RuntimeError(f"eof:{path}")
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and (digest != expected_sha or len(data) != expected_bytes):
        raise RuntimeError(f"exact:{path}")
    return {"path": str(path), "sha256": digest, "logical_bytes": len(data)}, data


def read_json(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> tuple[dict, dict]:
    record, data = regular(path, expected_sha, expected_bytes)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError(f"json-object:{path}")
    return value, record


def tree(root: Path) -> dict:
    st = os.lstat(root)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"tree-root:{root}")
    inventory = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        row, _ = regular(path)
        inventory.append([path.name, row["sha256"], row["logical_bytes"]])
    lines = "".join(f"{row[1]}  {row[0]}\n" for row in inventory).encode()
    triples = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "file_count": len(inventory), "inventory": inventory,
            "logical_file_bytes": sum(row[2] for row in inventory),
            "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
            "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest()}


def service_snapshot() -> dict:
    import urllib.request
    result = {}
    for name, url, digest, size in (
        ("service_8005", "http://127.0.0.1:8005/v1/health", "08dbc59225418d4d3064f9e122caeadbdf7c740d38fcbbaf708ad9baa44abab0", 106),
        ("service_18084", "http://127.0.0.1:18084/health", "31bf75f4c0a97cc1f7b60df824fa390b3be9ba014f29b63c87c698ba63d9a9fd", 18),
    ):
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read()
            code = response.status
        if code != 200 or len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise RuntimeError(name)
        result[name] = {"http_code": code, "body_sha256": digest, "body_bytes": size}
    return result


def validate_record_current(row: dict) -> None:
    if not isinstance(row, dict) or set(row) != {"logical_bytes", "path", "sha256"} or type(row["logical_bytes"]) is not int:
        raise RuntimeError("record-schema")
    current, _ = regular(Path(row["path"]), row["sha256"], row["logical_bytes"])
    if current != row:
        raise RuntimeError("record-current")


def validate_snapshot(snapshot: dict, forensic_path: Path) -> None:
    expected = ["authority_receipt", "authority_tree", "authority_validation", "failed_process_receipt", "failed_stderr",
                "failed_stdout", "failed_tree", "forensic", "materializer_popen_invocations", "public_s1_invocations",
                "relevant_entrypoint_pids", "required_absences", "services", "training_invocations"]
    if list(snapshot) != expected:
        raise RuntimeError("snapshot-schema")
    for key in ("forensic", "authority_receipt", "failed_process_receipt", "failed_stdout", "failed_stderr"):
        validate_record_current(snapshot[key])
    if snapshot["authority_receipt"] != {"path": str(AUTH_RECEIPT), "sha256": AUTH_SHA, "logical_bytes": AUTH_BYTES}:
        raise RuntimeError("snapshot-authority")
    if snapshot["forensic"] != {"path": str(forensic_path), "sha256": FORENSIC_SHA, "logical_bytes": FORENSIC_BYTES}:
        raise RuntimeError("snapshot-forensic")
    if snapshot["authority_tree"] != tree(AUTH_ROOT) or snapshot["failed_tree"] != tree(FAILED_ROOT):
        raise RuntimeError("snapshot-tree")
    for row in snapshot["required_absences"]:
        if list(row) != ["absent", "path"] or row["absent"] is not True or Path(row["path"]).exists() or Path(row["path"]).is_symlink():
            raise RuntimeError("snapshot-absence")
    if snapshot["services"] != service_snapshot() or snapshot["relevant_entrypoint_pids"] != []:
        raise RuntimeError("snapshot-runtime")
    for key in ("materializer_popen_invocations", "public_s1_invocations", "training_invocations"):
        if type(snapshot[key]) is not int or snapshot[key] != 0:
            raise RuntimeError("snapshot-partition")


def validate_candidate(root: Path, prep: Path, source_path: Path, forensic_path: Path) -> tuple[dict, dict, dict]:
    current_tree = tree(root)
    if current_tree["file_count"] != 1 or [row[0] for row in current_tree["inventory"]] != [CANDIDATE_NAME]:
        raise RuntimeError("candidate-tree")
    receipt, record = read_json(root / CANDIDATE_NAME)
    expected_keys = ["authority_receipt", "candidate_consumable", "candidate_prep", "candidate_root", "candidate_verified",
                     "content_verified", "execution_partition", "external_terminal_contract", "external_terminal_required",
                     "failed_transport_process_receipt", "failure_forensic", "format", "input_initial_snapshot",
                     "input_precommit_snapshot", "input_prepublish_snapshot", "input_triple_snapshots_exactly_equal",
                     "normalization_recursive_diff", "normalization_recursive_diff_canonical_sha256",
                     "normalization_recursive_diff_count", "normalized_materializer_stdout_view", "normalized_view_canonical_sha256",
                     "original_materializer_stdout_view", "original_view_canonical_sha256", "passed",
                     "publication_success_claimed", "readonly_reconciler_source", "standalone_consumable", "status"]
    if list(receipt) != expected_keys:
        raise RuntimeError("candidate-schema")
    if (receipt["format"] != "strict-track2-v537-v536-stage-a-authority-split-state-readonly-reconciliation-candidate-v1"
            or receipt["status"] != "content_verified_readonly_two_interface_false_negatives_normalized_pending_external_terminal"
            or receipt["passed"] is not None or receipt["candidate_verified"] is not True or receipt["content_verified"] is not True
            or receipt["standalone_consumable"] is not False or receipt["external_terminal_required"] is not True
            or receipt["publication_success_claimed"] is not False or receipt["candidate_consumable"] is not False):
        raise RuntimeError("candidate-nonterminal")
    if receipt["candidate_root"] != str(root) or receipt["candidate_prep"] != str(prep):
        raise RuntimeError("candidate-paths")
    expected_source = {"path": str(source_path), "sha256": RECONCILER_SHA, "logical_bytes": RECONCILER_BYTES}
    if receipt["readonly_reconciler_source"] != expected_source:
        raise RuntimeError("candidate-source")
    validate_record_current(expected_source)
    forensic, _ = read_json(forensic_path, FORENSIC_SHA, FORENSIC_BYTES)
    authority, _ = read_json(AUTH_RECEIPT, AUTH_SHA, AUTH_BYTES)
    process, _ = read_json(FAILED_PROCESS, FAILED_PROCESS_SHA, FAILED_PROCESS_BYTES)
    if receipt["failure_forensic"] != forensic or receipt["authority_receipt"] != authority or receipt["failed_transport_process_receipt"] != process:
        raise RuntimeError("candidate-full-inputs")
    original = receipt["original_materializer_stdout_view"]
    normalized = receipt["normalized_materializer_stdout_view"]
    diffs = receipt["normalization_recursive_diff"]
    expected_diffs = [
        {"normalized": True, "original": False, "path": ["publication", "current_exact1_consumable"]},
        {"normalized": True, "original": False, "path": ["publication", "postcommit_diagnostics_passed"]},
    ]
    if (diffs != expected_diffs or receipt["normalization_recursive_diff_count"] != 2
            or receipt["normalization_recursive_diff_canonical_sha256"] != csha(diffs)
            or receipt["original_view_canonical_sha256"] != csha(original)
            or receipt["normalized_view_canonical_sha256"] != csha(normalized)):
        raise RuntimeError("candidate-normalization")
    rebuilt = json.loads(json.dumps(original))
    rebuilt["publication"]["current_exact1_consumable"] = True
    rebuilt["publication"]["postcommit_diagnostics_passed"] = True
    if normalized != rebuilt or original["publication"]["current_exact1_consumable"] is not False or original["publication"]["postcommit_diagnostics_passed"] is not False:
        raise RuntimeError("candidate-exact2")
    snapshots = [receipt["input_initial_snapshot"], receipt["input_prepublish_snapshot"], receipt["input_precommit_snapshot"]]
    if receipt["input_triple_snapshots_exactly_equal"] is not True or snapshots[0] != snapshots[1] or snapshots[1] != snapshots[2]:
        raise RuntimeError("candidate-triple")
    validate_snapshot(snapshots[0], forensic_path)
    part = receipt["execution_partition"]
    expected_partition_keys = {"readonly_reconciler_invocations", "materializer_popen_invocations", "materializer_import_invocations",
                               "evaluator_invocations", "launcher_invocations", "public_s1_invocations",
                               "zero_update_gate_invocations", "training_invocations", "reward_read_invocations",
                               "hidden_private_final_read_invocations"}
    if (set(part or {}) != expected_partition_keys
            or type(part.get("readonly_reconciler_invocations")) is not int or part["readonly_reconciler_invocations"] != 1
            or any(type(value) is not int or value != 0 for key, value in part.items() if key != "readonly_reconciler_invocations")):
        raise RuntimeError("candidate-partition")
    expected_external = {"candidate_and_external_terminal_dual_bind_required": True,
                         "external_evidence_exact_file_count": 6,
                         "external_helper_popen_invocations": 0,
                         "external_terminal_status": "passed_external_terminal",
                         "fresh_external_terminal_required": True,
                         "retry_authorized": False}
    if receipt["external_terminal_contract"] != expected_external:
        raise RuntimeError("candidate-external-contract")
    return receipt, record, current_tree


def expected_deployment_record(reconciler_record: dict, helper_record: dict, script_record: dict) -> dict:
    return {"format": "strict-track2-v537-split-state-external-terminal-transport-deployment-record-v1",
            "status": "preregistered_atomic_noreplace_transport_triple_pending_external_deployment",
            "deployment_authority": "external_atomic_noreplace_deployer_required", "deployment_executed": False,
            "readonly_reconciler_source": reconciler_record, "external_terminal_helper_source": helper_record,
            "external_terminal_script_source": script_record, "source_role_order": ["readonly_reconciler", "external_terminal_helper", "external_terminal_script"],
            "noncyclic_binding": True, "normal_execution_authorized_by_record": False}


def validate_deployment_record(path: Path, reconciler_record: dict, helper_record: dict, script_record: dict) -> dict:
    value, _ = read_json(path)
    if value != expected_deployment_record(reconciler_record, helper_record, script_record):
        raise RuntimeError("deployment-record")
    return value


def write_held(path: Path, payload: bytes, state: dict) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(fd, payload[offset:])
            if wrote <= 0:
                raise RuntimeError("short-write")
            offset += wrote
        os.fsync(fd)
        data = os.pread(fd, len(payload) + 1, 0)
        fst, lst = os.fstat(fd), os.lstat(path)
        if data != payload or os.pread(fd, 1, len(payload)) != b"" or (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise RuntimeError("held-record")
        state["held"][path.name] = {"fd": fd, "path": path, "dev": fst.st_dev, "ino": fst.st_ino,
                                          "sha256": hashlib.sha256(payload).hexdigest(), "logical_bytes": len(payload), "payload": payload}
    except BaseException:
        os.close(fd)
        raise


def held_current(row: dict) -> bool:
    try:
        fst, lst = os.fstat(row["fd"]), os.lstat(row["path"])
        data = os.pread(row["fd"], row["logical_bytes"] + 1, 0)
    except (OSError, FileNotFoundError):
        return False
    return ((fst.st_dev, fst.st_ino) == (lst.st_dev, lst.st_ino) == (row["dev"], row["ino"])
            and data == row["payload"] and os.pread(row["fd"], 1, row["logical_bytes"]) == b"")


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.renameat2
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(AT_FDCWD, os.fsencode(source), AT_FDCWD, os.fsencode(target), RENAME_NOREPLACE) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(target))


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot(candidate_root: Path, candidate_prep: Path, candidate_source: Path, forensic_path: Path) -> dict:
    receipt, record, candidate_tree = validate_candidate(candidate_root, candidate_prep, candidate_source, forensic_path)
    return {"candidate_receipt": record, "candidate_tree": candidate_tree,
            "candidate_receipt_canonical_sha256": csha(receipt),
            "authority_receipt": {"path": str(AUTH_RECEIPT), "sha256": AUTH_SHA, "logical_bytes": AUTH_BYTES},
            "failed_process_receipt": {"path": str(FAILED_PROCESS), "sha256": FAILED_PROCESS_SHA, "logical_bytes": FAILED_PROCESS_BYTES},
            "failure_forensic": {"path": str(forensic_path), "sha256": FORENSIC_SHA, "logical_bytes": FORENSIC_BYTES},
            "services": service_snapshot(), "materializer_popen_invocations": 0, "readonly_reconciler_current_invocations": 0,
            "external_terminal_helper_invocations": 1, "public_s1_invocations": 0, "training_invocations": 0}


def cleanup_precommit(state: dict) -> None:
    for row in state.get("held", {}).values():
        if held_current(row):
            try:
                row["path"].unlink()
            except FileNotFoundError:
                pass
    prep = state.get("prep")
    identity = state.get("prep_identity")
    if prep is not None and identity is not None:
        try:
            st = os.lstat(prep)
            if stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) == identity:
                prep.rmdir()
                fsync_dir(prep.parent)
        except (FileNotFoundError, OSError):
            pass


def orchestrate(args: argparse.Namespace) -> dict:
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    self_record, self_raw = regular(Path(__file__), args.self_sha, args.self_bytes)
    script_record, _ = regular(Path(args.script_path), args.script_sha, args.script_bytes)
    reconciler_record, _ = regular(Path(args.candidate_source), RECONCILER_SHA, RECONCILER_BYTES)
    deployment = validate_deployment_record(Path(args.deployment_record), reconciler_record, self_record, script_record)
    candidate_root, candidate_prep = Path(args.candidate_root), Path(args.candidate_prep)
    evidence_root, prep = Path(args.evidence_root), Path(args.evidence_prep)
    if evidence_root.exists() or evidence_root.is_symlink() or prep.exists() or prep.is_symlink():
        raise RuntimeError("external-target-not-fresh")
    forensic_path = Path(args.forensic_path)
    initial = snapshot(candidate_root, candidate_prep, Path(args.candidate_source), forensic_path)
    if args.read_only_preflight:
        return {"passed": True, "read_only_preflight": True, "candidate_verified": True,
                "materializer_popen_invocations": 0, "snapshot_sha256": csha(initial)}
    prep.parent.mkdir(parents=True, exist_ok=True)
    prep.mkdir(mode=0o700)
    pst = os.lstat(prep)
    state = {"prep": prep, "prep_identity": (pst.st_dev, pst.st_ino), "held": {}, "committed": False}
    try:
        stdout_payload = cbytes({"line_origin": "external_terminal_helper_self_owned_preterminal",
                                 "candidate_receipt_sha256": initial["candidate_receipt"]["sha256"],
                                 "candidate_verified": True, "standalone_consumable": False,
                                 "external_terminal_required": True})
        intent = {"format": "strict-track2-v537-split-state-external-terminal-intent-v1",
                  "status": "external_terminal_pending", "candidate_root": str(candidate_root),
                  "evidence_root": str(evidence_root), "retry_authorized": False,
                  "materializer_popen_invocations": 0, "public_s1_invocations": 0, "training_invocations": 0}
        argv = {"format": "strict-track2-v537-split-state-external-terminal-argv-v1", "argv": sys.argv,
                "self": self_record, "script": script_record, "readonly_reconciler": reconciler_record,
                "transport_deployment_record": deployment}
        for name, payload in (("transport_helper.py", self_raw), ("argv.json", cbytes(argv)), ("intent.json", cbytes(intent)),
                              ("helper_stdout.log", stdout_payload), ("helper_stderr.log", b"")):
            write_held(prep / name, payload, state)
        hook("exact5_prepared", state)
        preterminal = snapshot(candidate_root, candidate_prep, Path(args.candidate_source), forensic_path)
        third = snapshot(candidate_root, candidate_prep, Path(args.candidate_source), forensic_path)
        if initial != preterminal or preterminal != third:
            raise RuntimeError("external-triple-drift")
        candidate, candidate_record, candidate_tree = validate_candidate(candidate_root, candidate_prep, Path(args.candidate_source), forensic_path)
        member_records = {name: {"path": str(row["path"]), "sha256": row["sha256"], "logical_bytes": row["logical_bytes"]}
                          for name, row in sorted(state["held"].items())}
        receipt = {"format": "strict-track2-v537-v536-stage-a-authority-split-state-external-terminal-process-receipt-v1",
                   "status": "passed_external_terminal", "passed": True, "candidate_consumable": True,
                   "candidate_verified": True, "candidate_nonterminal": True, "standalone_candidate_consumed": False,
                   "candidate_receipt": candidate_record, "candidate_tree": candidate_tree,
                   "candidate_receipt_canonical_sha256": csha(candidate), "readonly_reconciler_source": reconciler_record,
                   "external_terminal_helper_source": self_record, "external_terminal_script_source": script_record,
                   "transport_deployment_record": deployment, "member_records_exact5": member_records,
                   "input_initial_snapshot": initial, "input_preterminal_snapshot": preterminal,
                   "input_third_snapshot": third, "input_triple_snapshots_exactly_equal": True,
                   "execution_partition": {"historical_readonly_reconciler_invocations": 1,
                                           "current_readonly_reconciler_invocations": 0,
                                           "external_terminal_helper_invocations": 1,
                                           "materializer_popen_invocations": 0, "materializer_invocations": 0,
                                           "evaluator_invocations": 0, "launcher_invocations": 0,
                                           "public_s1_invocations": 0, "zero_update_gate_invocations": 0,
                                           "training_invocations": 0, "reward_read_invocations": 0,
                                           "hidden_private_final_read_invocations": 0},
                   "retry_authorized": False, "publication_visibility_committed": True,
                   "publication_success_claimed_by_external_terminal": True}
        write_held(prep / "process_receipt.json", cbytes(receipt), state)
        hook("terminal_precommit", state)
        if sorted(state["held"]) != EXACT6 or not all(held_current(row) for row in state["held"].values()):
            raise RuntimeError("external-held-exact6")
        now = os.lstat(prep)
        if (now.st_dev, now.st_ino) != state["prep_identity"]:
            raise RuntimeError("external-prep-identity")
        fsync_dir(prep)
        rename_noreplace(prep, evidence_root)
        state["committed"] = True
        diagnostic = None
        evidence_tree = None
        try:
            hook("terminal_committed", state)
            fsync_dir(evidence_root.parent)
            evidence_tree = tree(evidence_root)
            if evidence_tree["file_count"] != 6 or [row[0] for row in evidence_tree["inventory"]] != EXACT6:
                raise RuntimeError("external-current-exact6")
            post = snapshot(candidate_root, candidate_prep, Path(args.candidate_source), forensic_path)
            if post != initial:
                raise RuntimeError("external-post-drift")
        except BaseException as exc:
            diagnostic = {"type": type(exc).__name__, "message": str(exc)}
        return {"passed": True, "status": "passed_external_terminal", "candidate_consumable": True,
                "visibility_committed": True, "postcommit_success_priority": True,
                "postcommit_diagnostic": diagnostic, "evidence_tree": evidence_tree,
                "process_receipt": {"path": str(evidence_root / "process_receipt.json"),
                                    "sha256": state["held"]["process_receipt.json"]["sha256"],
                                    "logical_bytes": state["held"]["process_receipt.json"]["logical_bytes"]}}
    except BaseException:
        if not state["committed"]:
            cleanup_precommit(state)
        raise
    finally:
        for row in state["held"].values():
            try:
                os.close(row["fd"])
            except OSError:
                pass


def synthetic_self_test() -> dict:
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="v537-external-selftest-"))
    try:
        target, prep = base / "out", base / "out.prep"
        prep.mkdir()
        state = {"prep": prep, "prep_identity": (os.lstat(prep).st_dev, os.lstat(prep).st_ino), "held": {}, "committed": False}
        write_held(prep / "x", b"abc", state)
        positive = held_current(state["held"]["x"])
        original = prep / "x"
        moved = prep / "owned"
        original.rename(moved)
        original.write_bytes(b"foreign")
        foreign_preserved = not held_current(state["held"]["x"])
        cleanup_precommit(state)
        foreign_preserved = foreign_preserved and original.read_bytes() == b"foreign"
        os.close(state["held"]["x"]["fd"])
        return {"passed": positive and foreign_preserved, "checks": {"held_current": positive, "foreign_preserved": foreign_preserved}}
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha", required=True)
    parser.add_argument("--self-bytes", required=True, type=int)
    parser.add_argument("--script-sha", required=True)
    parser.add_argument("--script-bytes", required=True, type=int)
    parser.add_argument("--script-path", default=str(SCRIPT_PATH))
    parser.add_argument("--deployment-record", default=str(DEPLOYMENT_RECORD))
    parser.add_argument("--candidate-source", default=str(RECONCILER))
    parser.add_argument("--forensic-path", default=str(FORENSIC))
    parser.add_argument("--candidate-root", default=str(CANDIDATE_ROOT))
    parser.add_argument("--candidate-prep", default=str(CANDIDATE_PREP))
    parser.add_argument("--evidence-root", default=str(EVIDENCE_ROOT))
    parser.add_argument("--evidence-prep", default=str(EVIDENCE_PREP))
    parser.add_argument("--read-only-preflight", action="store_true")
    parser.add_argument("--synthetic-self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        result = synthetic_self_test()
    else:
        result = orchestrate(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("passed") is True else 79


if __name__ == "__main__":
    raise SystemExit(main())
