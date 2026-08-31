#!/usr/bin/env python3
"""Read-only reconciliation of the v536 Stage-A split state.

This program never imports or executes the materializer, evaluator, launcher, or
training code.  It independently validates the already-visible authority and
the failed transport evidence, normalizes exactly two false-negative booleans,
and publishes a nonterminal candidate for a separate external terminalizer.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import stat
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
S = ROOT / "pipeline/scripts"
SELF_PATH = S / "reconcile_v537_v536_stage_a_authority_split_state_readonly.py"
FORENSIC = S / "v537_v536_stage_a_authority_split_state_failure_forensic.json"
FORENSIC_SHA = "8a8abe3b18ac67b7c6739f2bab7d561d14755d327174c811ca887aeb37aa1c2f"
FORENSIC_BYTES = 13186
AUTH_ROOT = J / "v535_v534_public_s1_zero_update_gate_execution_authority_seed1668_20260827"
AUTH_PREP = AUTH_ROOT.with_name(AUTH_ROOT.name + ".authority-prep")
AUTH_RECEIPT = AUTH_ROOT / "authority_receipt.json"
AUTH_SHA = "2b7f1c1e82cccb107da8e44af883940701f2427deccef75749e3afc2d85d7358"
AUTH_BYTES = 125125
AUTH_LINES_SHA = "06913cd19f19ca2087cc631fdd4895e73fe946ece01d813ba654cc8802acfb23"
AUTH_CANON_SHA = "b8e96449ce050d40147603153cb49bc1c3e05c65f1b31fbbe4351ff9620c44fc"
FAILED_ROOT = J / "v536_v535_stage_a_contract_absence_schema_repair_authority_materialization_evidence_seed1668_20260827"
FAILED_PREP = FAILED_ROOT.with_name(FAILED_ROOT.name + ".execution-prep")
FAILED_PROCESS = FAILED_ROOT / "process_receipt.json"
FAILED_PROCESS_SHA = "0689907a3e08721893b2a73e737a0421b1cb0c7ed111c8dd6769daa75f4c6b92"
FAILED_PROCESS_BYTES = 8324
FAILED_STDOUT = FAILED_ROOT / "materializer_stdout.log"
FAILED_STDOUT_SHA = "8c8c51ac9ab4668461640acc527dbe4a3db30890ac7d6c12a5dc43c872bcee33"
FAILED_STDOUT_BYTES = 496
FAILED_STDERR = FAILED_ROOT / "materializer_stderr.log"
FAILED_TREE_LINES = "a6c03548cb43be71c489c9712bb97bdcd0ff06a642df0a791cb91e03b0209d14"
FAILED_TREE_CANON = "73445fc5a2f7d54c7ece8e6d093f78b8d0b896c3ea33401499985cacf8b2a6a9"
FAILED_TREE_BYTES = 112792
FAILED_NAMES = ["argv.json", "intent.json", "materializer_stderr.log", "materializer_stdout.log", "process_receipt.json", "transport_helper.py"]
CANDIDATE_ROOT = J / "v537_v536_stage_a_authority_split_state_readonly_candidate_seed1669_20260827"
CANDIDATE_PREP = CANDIDATE_ROOT.with_name(CANDIDATE_ROOT.name + ".candidate-prep")
CANDIDATE_NAME = "reconciliation_candidate_receipt.json"
ATTEMPT = J / "v535_v534_public_s1_zero_update_gate_attempt_seed1668_20260827"
OUTPUT = Path("/root/v535_v534_public_s1_zero_update_gate_seed1668_20260827")
EXTERNAL_ROOT = J / "v537_v536_stage_a_authority_split_state_external_terminal_evidence_seed1669_20260827"
EXTERNAL_PREP = EXTERNAL_ROOT.with_name(EXTERNAL_ROOT.name + ".evidence-prep")
AT_FDCWD = -100
RENAME_NOREPLACE = 1


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
        size = fst.st_size
        data = os.pread(fd, size + 1, 0)
        if len(data) != size or os.pread(fd, 1, size) != b"":
            raise RuntimeError(f"eof:{path}")
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and (digest != expected_sha or size != expected_bytes):
        raise RuntimeError(f"exact:{path}")
    return {"path": str(path), "sha256": digest, "logical_bytes": size}, data


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
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        record, _ = regular(path)
        inventory.append([path.name, record["sha256"], record["logical_bytes"]])
    lines = "".join(f"{row[1]}  {row[0]}\n" for row in inventory).encode()
    triples = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "file_count": len(inventory), "inventory": inventory,
            "logical_file_bytes": sum(row[2] for row in inventory),
            "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
            "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest()}


def validate_authority(authority: dict) -> dict:
    if len(authority) != 62 or authority.get("passed") is not True:
        raise RuntimeError("authority-top")
    if authority.get("format") != "strict-track2-v535-v534-public-s1-zero-update-gate-execution-authority-v1":
        raise RuntimeError("authority-format")
    if authority.get("status") != "authorized_exact_one_public_s1_zero_update_boundary_pending_external_execution":
        raise RuntimeError("authority-status")
    checks = authority.get("checks")
    check_keys = authority.get("check_keys")
    if not isinstance(checks, dict) or len(checks) != 31 or list(checks) != check_keys or any(type(value) is not bool or value is not True for value in checks.values()):
        raise RuntimeError("authority-checks")
    order = authority.get("source_role_order")
    closure = authority.get("source_closure")
    aliases = authority.get("source_aliases")
    if not isinstance(order, list) or len(order) != 31 or len(set(order)) != 31:
        raise RuntimeError("authority-order")
    if not isinstance(closure, dict) or set(closure) != set(order) or set(aliases or {}) != set(order):
        raise RuntimeError("authority-closure")
    for role in order:
        row = closure[role]
        if list(row) != ["logical_bytes", "path", "sha256"] or type(row["logical_bytes"]) is not int:
            raise RuntimeError(f"authority-row:{role}")
        current, _ = regular(Path(row["path"]), row["sha256"], row["logical_bytes"])
        if current != row:
            raise RuntimeError(f"authority-current:{role}")
        alias = aliases[role]
        if authority.get(alias) != row:
            raise RuntimeError(f"authority-alias:{role}")
    authorization = authority.get("authorization")
    if (type(authorization.get("public_s1_zero_update_boundary_invocations_authorized")) is not int
            or authorization["public_s1_zero_update_boundary_invocations_authorized"] != 1
            or type(authorization.get("public_s1_zero_update_boundary_invocations_consumed")) is not int
            or authorization["public_s1_zero_update_boundary_invocations_consumed"] != 0
            or authorization.get("retry_authorized") is not False
            or authorization.get("training_authorized") is not False):
        raise RuntimeError("authority-authorization")
    if authority.get("input_snapshots_exactly_equal") is not True or authority.get("input_pre_snapshot") != authority.get("input_post_snapshot"):
        raise RuntimeError("authority-snapshots")
    return {"top_key_count": 62, "check_count": 31, "source_count": 31,
            "source_aliases_exact": True, "source_closure_current": True,
            "authorized": 1, "consumed": 0, "prepost_equal": True}


def validate_failed(process: dict, stdout_raw: bytes) -> tuple[dict, dict, list[dict]]:
    if (process.get("passed") is not False or process.get("status") != "failed_no_retry"
            or process.get("error") != "materializer native stdout must be empty"):
        raise RuntimeError("failed-process")
    part = process
    if (type(part.get("transport_helper_invocations")) is not int or part.get("transport_helper_invocations") != 1
            or type(part.get("authority_materializer_invocations")) is not int or part.get("authority_materializer_invocations") != 1):
        raise RuntimeError("failed-partition")
    for key in ("public_s1_evaluator_invocations", "public_s1_zero_update_boundary_invocations", "zero_update_gate_invocations", "training_invocations"):
        if type(part.get(key)) is not int or part[key] != 0:
            raise RuntimeError(f"failed-partition:{key}")
    if stdout_raw.count(b"\n") != 1 or not stdout_raw.endswith(b"\n"):
        raise RuntimeError("stdout-lines")
    original = json.loads(stdout_raw)
    publication = original.get("publication")
    if (original.get("passed") is not True or publication.get("visibility_committed") is not True
            or publication.get("postcommit_diagnostics") != [] or publication.get("retry_authorized") is not False
            or publication.get("current_exact1_consumable") is not False
            or publication.get("postcommit_diagnostics_passed") is not False
            or original.get("receipt_sha256") != AUTH_SHA or original.get("receipt_bytes") != AUTH_BYTES):
        raise RuntimeError("stdout-original")
    normalized = json.loads(json.dumps(original))
    normalized["publication"]["current_exact1_consumable"] = True
    normalized["publication"]["postcommit_diagnostics_passed"] = True
    diffs = [
        {"path": ["publication", "current_exact1_consumable"], "original": False, "normalized": True},
        {"path": ["publication", "postcommit_diagnostics_passed"], "original": False, "normalized": True},
    ]
    return original, normalized, diffs


def services() -> dict:
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


def relevant_pids() -> list[dict]:
    needles = {str(SELF_PATH), "materialize_v535_v534_public_s1_zero_update_gate_execution_authority.py",
               "evaluate_v535_v534_public_s1_zero_update_gate.py", "launch_v535_v534_public_s1_zero_update_gate.py"}
    own = os.getpid()
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            decoded = [item.decode(errors="surrogateescape") for item in argv if item]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(item in needles for item in decoded):
            rows.append({"pid": int(proc.name), "argv": decoded})
    if rows:
        raise RuntimeError("relevant-pids")
    return rows


def validate_forensic(value: dict) -> None:
    if (value.get("format") != "strict-track2-v537-v536-stage-a-authority-split-state-failure-forensic-v1"
            or value.get("status") != "frozen_split_state_failed_no_retry_pending_fresh_readonly_reconciliation_external_terminal"):
        raise RuntimeError("forensic-schema")
    boundary = value.get("fresh_reconciliation_boundary")
    expected_true = ["actual_exact1_must_be_independently_recomputed", "candidate_and_external_terminal_dual_bind_required",
                     "fresh_external_terminal_evidence_required", "fresh_readonly_reconciler_required",
                     "full_authority_receipt_tree_source_alias_check_revalidation_required",
                     "native_stdout_exact1_json_line_revalidation_required", "old_false_consumability_claim_must_be_preserved_as_observed",
                     "success_only_via_fresh_external_terminal"]
    expected_false = ["canonical_deployment_authorized_by_this_forensic", "old_authority_overwrite_or_rematerialization_authorized",
                      "old_failed_evidence_cleanup_or_retry_authorized", "old_materializer_rerun_authorized",
                      "public_s1_zero_update_execution_authorized_by_this_forensic", "retry_authorized"]
    if any(boundary.get(key) is not True for key in expected_true) or any(boundary.get(key) is not False for key in expected_false):
        raise RuntimeError("forensic-boundary")
    if type(boundary.get("fresh_reconciler_materializer_popen_invocations_authorized")) is not int or boundary["fresh_reconciler_materializer_popen_invocations_authorized"] != 0:
        raise RuntimeError("forensic-popen")


def snapshot(forensic_path: Path = FORENSIC) -> tuple[dict, dict, dict, dict, list[dict]]:
    forensic, forensic_record = read_json(forensic_path, FORENSIC_SHA, FORENSIC_BYTES)
    validate_forensic(forensic)
    authority, authority_record = read_json(AUTH_RECEIPT, AUTH_SHA, AUTH_BYTES)
    authority_validation = validate_authority(authority)
    authority_tree = tree(AUTH_ROOT)
    if (authority_tree["file_count"] != 1 or authority_tree["inventory"] != [["authority_receipt.json", AUTH_SHA, AUTH_BYTES]]
            or authority_tree["canonical_json_triples_digest_sha256"] != AUTH_CANON_SHA):
        raise RuntimeError("authority-tree")
    process, process_record = read_json(FAILED_PROCESS, FAILED_PROCESS_SHA, FAILED_PROCESS_BYTES)
    stdout_record, stdout_raw = regular(FAILED_STDOUT, FAILED_STDOUT_SHA, FAILED_STDOUT_BYTES)
    stderr_record, stderr_raw = regular(FAILED_STDERR, hashlib.sha256(b"").hexdigest(), 0)
    if stderr_raw != b"":
        raise RuntimeError("stderr")
    failed_tree = tree(FAILED_ROOT)
    if (failed_tree["file_count"] != 6 or [row[0] for row in failed_tree["inventory"]] != FAILED_NAMES
            or failed_tree["logical_file_bytes"] != FAILED_TREE_BYTES
            or failed_tree["canonical_json_triples_digest_sha256"] != FAILED_TREE_CANON):
        raise RuntimeError("failed-tree")
    original, normalized, diffs = validate_failed(process, stdout_raw)
    absences = [AUTH_PREP, FAILED_PREP, ATTEMPT, ATTEMPT.with_name(ATTEMPT.name + ".attempt-prep"), OUTPUT,
                OUTPUT.with_name(OUTPUT.name + ".output-prep"), EXTERNAL_ROOT, EXTERNAL_PREP]
    if any(path.exists() or path.is_symlink() for path in absences):
        raise RuntimeError("required-absence")
    snap = {"forensic": forensic_record, "authority_receipt": authority_record, "authority_tree": authority_tree,
            "authority_validation": authority_validation, "failed_process_receipt": process_record,
            "failed_stdout": stdout_record, "failed_stderr": stderr_record, "failed_tree": failed_tree,
            "required_absences": [{"path": str(path), "absent": True} for path in absences],
            "services": services(), "relevant_entrypoint_pids": relevant_pids(),
            "materializer_popen_invocations": 0, "public_s1_invocations": 0, "training_invocations": 0}
    return snap, authority, process, original, [normalized, diffs, forensic]


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


def publish(receipt: dict, root: Path, prep: Path) -> dict:
    if root.exists() or root.is_symlink() or prep.exists() or prep.is_symlink():
        raise RuntimeError("candidate-target-not-fresh")
    prep.parent.mkdir(parents=True, exist_ok=True)
    prep.mkdir(mode=0o700)
    pst = os.lstat(prep)
    member = prep / CANDIDATE_NAME
    fd = -1
    record = None
    committed = False
    diagnostic = None
    try:
        payload = cbytes(receipt)
        fd = os.open(member, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        offset = 0
        while offset < len(payload):
            wrote = os.write(fd, payload[offset:])
            if wrote <= 0:
                raise RuntimeError("candidate-short-write")
            offset += wrote
        os.fsync(fd)
        data = os.pread(fd, len(payload) + 1, 0)
        fst = os.fstat(fd)
        lst = os.lstat(member)
        if data != payload or os.pread(fd, 1, len(payload)) != b"" or (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
            raise RuntimeError("candidate-held-record")
        record = {"path": str(member), "sha256": hashlib.sha256(payload).hexdigest(), "logical_bytes": len(payload), "dev": fst.st_dev, "ino": fst.st_ino}
        fsync_dir(prep)
        now = os.lstat(prep)
        if (now.st_dev, now.st_ino) != (pst.st_dev, pst.st_ino):
            raise RuntimeError("candidate-prep-identity")
        rename_noreplace(prep, root)
        committed = True
        try:
            fsync_dir(root.parent)
            current, _ = regular(root / CANDIDATE_NAME, record["sha256"], record["logical_bytes"])
            tree_now = tree(root)
        except BaseException as exc:
            diagnostic = {"type": type(exc).__name__, "message": str(exc)}
            current = {"path": str(root / CANDIDATE_NAME), "sha256": record["sha256"], "logical_bytes": record["logical_bytes"]}
            tree_now = None
        return {"candidate_visibility_committed": True, "publication_success_claimed": False,
                "standalone_consumable": False, "external_terminal_required": True,
                "postcommit_diagnostic": diagnostic, "receipt": current, "tree": tree_now}
    except BaseException:
        if not committed:
            if fd >= 0 and record is not None:
                try:
                    lst = os.lstat(member)
                    fst = os.fstat(fd)
                    if (lst.st_dev, lst.st_ino) == (fst.st_dev, fst.st_ino) == (record["dev"], record["ino"]):
                        member.unlink()
                except FileNotFoundError:
                    pass
            try:
                now = os.lstat(prep)
                if stat.S_ISDIR(now.st_mode) and (now.st_dev, now.st_ino) == (pst.st_dev, pst.st_ino):
                    prep.rmdir()
                    fsync_dir(prep.parent)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def build_receipt(self_record: dict, snapshots: list[dict], authority: dict, process: dict,
                  original: dict, normalized: dict, diffs: list[dict], forensic: dict,
                  root: Path, prep: Path) -> dict:
    if snapshots[0] != snapshots[1] or snapshots[1] != snapshots[2]:
        raise RuntimeError("triple-snapshot-drift")
    return {
        "format": "strict-track2-v537-v536-stage-a-authority-split-state-readonly-reconciliation-candidate-v1",
        "status": "content_verified_readonly_two_interface_false_negatives_normalized_pending_external_terminal",
        "passed": None, "candidate_verified": True, "content_verified": True,
        "standalone_consumable": False, "external_terminal_required": True,
        "publication_success_claimed": False, "candidate_consumable": False,
        "candidate_root": str(root), "candidate_prep": str(prep),
        "readonly_reconciler_source": self_record,
        "failure_forensic": forensic,
        "authority_receipt": authority,
        "failed_transport_process_receipt": process,
        "original_materializer_stdout_view": original,
        "normalized_materializer_stdout_view": normalized,
        "original_view_canonical_sha256": csha(original),
        "normalized_view_canonical_sha256": csha(normalized),
        "normalization_recursive_diff": diffs,
        "normalization_recursive_diff_count": 2,
        "normalization_recursive_diff_canonical_sha256": csha(diffs),
        "input_initial_snapshot": snapshots[0],
        "input_prepublish_snapshot": snapshots[1],
        "input_precommit_snapshot": snapshots[2],
        "input_triple_snapshots_exactly_equal": True,
        "execution_partition": {"readonly_reconciler_invocations": 1, "materializer_popen_invocations": 0,
                                "materializer_import_invocations": 0, "evaluator_invocations": 0,
                                "launcher_invocations": 0, "public_s1_invocations": 0,
                                "zero_update_gate_invocations": 0, "training_invocations": 0,
                                "reward_read_invocations": 0, "hidden_private_final_read_invocations": 0},
        "external_terminal_contract": {"fresh_external_terminal_required": True,
                                       "candidate_and_external_terminal_dual_bind_required": True,
                                       "external_terminal_status": "passed_external_terminal",
                                       "external_evidence_exact_file_count": 6,
                                       "external_helper_popen_invocations": 0,
                                       "retry_authorized": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha", required=True)
    parser.add_argument("--self-bytes", required=True, type=int)
    parser.add_argument("--candidate-root", default=str(CANDIDATE_ROOT))
    parser.add_argument("--candidate-prep", default=str(CANDIDATE_PREP))
    parser.add_argument("--forensic-path", default=str(FORENSIC))
    parser.add_argument("--read-only-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    self_record, _ = regular(Path(__file__), args.self_sha, args.self_bytes)
    root, prep = Path(args.candidate_root), Path(args.candidate_prep)
    forensic_path = Path(args.forensic_path)
    initial, authority, process, original, extra = snapshot(forensic_path)
    normalized, diffs, forensic = extra
    prepublish, *_ = snapshot(forensic_path)
    precommit, *_ = snapshot(forensic_path)
    receipt = build_receipt(self_record, [initial, prepublish, precommit], authority, process, original, normalized, diffs, forensic, root, prep)
    if args.read_only_preflight:
        print(json.dumps({"passed": True, "read_only_preflight": True, "materializer_popen_invocations": 0,
                          "normalization_recursive_diff_count": 2, "snapshot_sha256": csha(initial)}, sort_keys=True, separators=(",", ":")))
        return 0
    result = publish(receipt, root, prep)
    print(json.dumps({"passed": None, "candidate_verified": True, "content_verified": True,
                      "standalone_consumable": False, "external_terminal_required": True,
                      "publication_success_claimed": False, "publication": result}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
