#!/usr/bin/env python3
"""Atomically register one v490 readonly-repair authority; never execute r2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
CONTRACT_PATH = ROOT / "pipeline/scripts/v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-postregistration-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_poststatic_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-postregistration-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_readonly_reconciliation_source_repair_attempt"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
AUTHORITY_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_authority_seed1632_20260825"
ATTEMPT_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
STATIC_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SOURCE_ROLES = {
    "repair_preregistration", "repair_design_contract", "repair_formal_materializer",
    "reconciler_r2", "static_auditor", "execution_wrapper",
}
STATIC_TOP_KEYS = {
    "format", "status", "passed", "checks", "check_keys", "check_key_set_sha256",
    "checks_sha256", "repair_preregistration", "design_contract", "materializer_source",
    "reconciler_r2_source", "static_auditor_source", "source_closure",
    "source_closure_sha256", "f813_registration_tree", "v487_authority_tree",
    "v487_helper_evidence_tree", "failed_reconciliation_attempt_tree", "ast_diff_proof",
    "synthetic_evidence", "required_absences", "runtime_observation",
    "readonly_reconciliation_authorized", "training_authorized", "submission_authorized",
    "superseded_static_source", "failed_static_execution_tree", "failed_static_process_receipt",
    "superseded_static_absences", "v489_failed_static_source",
    "v489_failed_static_execution_tree", "v489_failed_static_process_receipt",
    "v489_superseded_static_absences", "v489_predeploy_transport_failure",
}
STATIC_CHECK_KEYS = sorted({
    "contract_current", "f813_tree_exact2", "failed_attempt_tree_exact4_no_retry",
    "failed_static_invocation_partition", "failed_static_process_no_retry",
    "failed_static_root_prep_absent", "failed_static_tree_exact6",
    "formal_authority_all_false", "formal_record_exact", "formal_runtime_no_execution",
    "formal_schema_exact21", "materializer_current", "no_live_process",
    "no_training_reward_outcome", "observed_exact7_current", "path_literal_ast_repair_exact",
    "predeploy_transport_failure_disclosed", "r2_current", "r2_exact4_strict_no_fallback",
    "r2_forbidden_imports_calls_absent", "r2_old_c71_dual_binding",
    "r2_snapshot_dual_ancestry", "real_publish_callgraph_fixture_passed",
    "second_failed_static_invocation_partition", "second_failed_static_process_no_retry",
    "second_failed_static_root_prep_absent", "second_failed_static_tree_exact6",
    "source_closure_digest_exact", "source_closure_exact8_current",
    "synthetic_tamper_suite_passed", "transparent_and_fresh_outputs_absent",
    "v487_authority_tree_exact1", "v487_helper_tree_exact5",
})
STATIC_KEYSET_SHA = "53b721e19caa9b6302986e65c78dac72e0026716d67852d1b886aba27492d7cf"
STATIC_CHECKS_SHA = "50edf45d3a0021c3054ff80b2fcc311ebb848272def1e0732aa3aaf7c2079a58"
AUTH_CHECK_KEYS = sorted({
    "authority_materializer_current", "current_absences", "execution_wrapper_current",
    "f813_exact2", "f88_failed_static_no_retry_partition",
    "f88_failed_static_source_current", "f88_failed_static_tree_exact6",
    "f88_superseded_static_roots_absent", "failed_attempt_exact4",
    "formal_evidence_exact6", "input_snapshots_equal", "no_live_process",
    "old_v487_no_retry", "postregistration_static_reg_exact1", "r2_current",
    "repair_formal_exact", "repair_formal_reg_exact1", "source_closure_current",
    "static_boundary_no_authority", "static_checks33_all_true",
    "static_execution_evidence_exact6", "static_receipt_exact", "transparent_absent",
    "v489_failed_static_no_retry_partition", "v489_failed_static_source_current",
    "v489_failed_static_tree_exact6", "v489_predeploy_transport_disclosed",
    "v489_superseded_static_roots_absent",
})
AUTH_KEYSET_SHA = "be4899694c2bf3c933c18896697a6e12b8227195b4be36185bdd94b9c10ea68f"
AUTH_CHECKS_SHA = "be4d8b6f375d8c522c11439593d7f7ea31a04f582239f6b0ffbf9bcd66c1212a"
AUTH_TOP_KEYS = {
    "format", "status", "passed", "authority_design_contract", "authority_materializer_source",
    "repair_preregistration", "repair_design_contract", "repair_formal_materializer",
    "reconciler_r2_source", "static_auditor_source", "execution_wrapper_source",
    "static_receipt", "repair_formal_registration_tree",
    "postregistration_static_registration_tree", "static_execution_evidence_tree",
    "f813_registration_tree", "failed_attempt_tree", "formal_materialization_evidence_tree",
    "attempt_root", "transparent_static_receipt_path", "historical_absences",
    "required_absences", "checks", "checks_sha256", "input_pre_snapshot",
    "input_post_snapshot", "input_snapshots_exactly_equal", "authorization",
    "runtime_observation", "superseded_static_source", "failed_static_execution_tree",
    "failed_static_process_receipt", "superseded_static_absences",
    "v489_failed_static_source", "v489_failed_static_execution_tree",
    "v489_failed_static_process_receipt", "v489_superseded_static_absences",
    "v489_predeploy_transport_failure",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_closure_sha256",
    "authority_materializer_source", "static_receipt", "static_receipt_contract",
    "static_failure_ancestry", "repair_formal_schema", "repair_formal_registration_tree",
    "postregistration_static_registration_tree", "static_execution_evidence_tree",
    "f813_registration_tree", "failed_attempt_tree", "formal_materialization_evidence_tree",
    "historical_absences", "current_absences_after_authority", "authority_receipt_contract",
    "execution_boundary",
}
AUTHORITY_SCHEMA_KEYS = {
    "format", "status", "top_keys", "check_keys", "check_key_set_sha256",
    "checks_sha256", "authorization_exact", "runtime_observation_exact",
}
STATIC_SCHEMA_KEYS = {
    "format", "status", "top_keys", "check_keys", "check_key_set_sha256",
    "checks_sha256", "runtime_observation_exact",
}
FORMAL_SCHEMA_KEYS = {
    "format", "status", "top_keys", "authorization_exact", "runtime_observation_exact",
}
FAILURE_ANCESTRY_KEYS = {
    "superseded_static_source", "failed_static_execution_tree",
    "failed_static_process_receipt", "superseded_static_absences",
    "v489_failed_static_source", "v489_failed_static_execution_tree",
    "v489_failed_static_process_receipt", "v489_superseded_static_absences",
    "v489_predeploy_transport_failure",
}
AUTHORIZATION = {
    "readonly_reconciliation_authorized": True, "attempts_authorized": 1,
    "attempts_consumed": 0, "retry_authorized": False,
    "phase_a_cache_qualification_authorized": False, "cache_reuse_authorized": False,
    "training_authorized": False, "folds_authorized": 0, "policy_updates": 0,
    "s1_authorized": False, "zero_update_authorized": False, "rl_authorized": False,
    "submission_authorized": False, "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME = {
    "authority_materialized": True, "static_audit_executed": True,
    "reconciler_r2_executed": False, "transparent_receipt_created": False,
    "phase_a_executed": False, "training_launched": False, "folds": 0,
    "policy_updates": 0,
}
STATIC_RUNTIME = {
    "static_audit_executed": True, "reconciler_r2_executed": False,
    "transparent_receipt_created": False, "phase_a_executed": False,
    "training_launched": False, "folds": 0, "policy_updates": 0,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True, "reconciler_r2_invocations": 0,
    "execution_wrapper_invocations": 0, "phase_a_invocations": 0,
    "training_invocations": 0, "reward_reads": 0, "dev_hidden_final_outcome_reads": 0,
}


class ControlledSignal(BaseException):
    pass


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pending(value) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(pending(item) for item in value.values())
    if isinstance(value, list):
        return any(pending(item) for item in value)
    return False


def regular(path: Path | str, digest: str, logical_bytes: int) -> dict:
    path = Path(path)
    if (path != path.resolve() or not path.is_file() or path.is_symlink()
            or sha(path) != digest or path.stat().st_size != logical_bytes):
        raise RuntimeError(f"file closure: {path}")
    return {"path": str(path), "sha256": digest, "logical_bytes": logical_bytes}


def record(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record schema")
    return regular(value["path"], value["sha256"], value["logical_bytes"])


def exact_tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {rel}\n" for rel, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "inventory": rows, "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def verify_tree(spec: dict) -> dict:
    expected = {"root", "inventory", "file_count", "logical_file_bytes",
                "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"}
    if not isinstance(spec, dict) or set(spec) != expected:
        raise RuntimeError("tree schema")
    actual = exact_tree(spec["root"])
    if actual != {key: spec[key] for key in actual}:
        raise RuntimeError(f"tree closure: {spec['root']}")
    return actual


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def directory_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("directory ownership")
    return stat.st_dev, stat.st_ino


def cleanup_owned_prep(path: Path, identity: tuple[int, int]) -> None:
    if not path.exists() or path.is_symlink() or directory_identity(path) != identity:
        return
    shutil.rmtree(path)
    fsync_dir(path.parent)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):
        raise FileExistsError(path)
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def publish_exact1(root: Path, prep: Path, receipt: dict, immutable_snapshot, phase_hook=None) -> dict:
    """Shared production/fixture publication path with owned cleanup and deferred signals."""
    if os.path.lexists(root) or os.path.lexists(prep):
        raise RuntimeError("publication prestate")
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
        prep_created = True
        prep_identity = directory_identity(prep)
        fsync_dir(prep.parent)
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline_mask)
            signals_blocked = False
        phase_hook("owned_prep_unblocked")
        atomic_json(prep / "authority_receipt.json", receipt)
        fsync_dir(prep)
        if immutable_snapshot() != immutable_pre:
            raise RuntimeError("post-write input drift")
        if directory_identity(prep) != prep_identity or os.path.lexists(root):
            raise RuntimeError("publication ownership drift")
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            signals_blocked = True
        phase_hook("commit_window_blocked")
        os.replace(prep, root)
        fsync_dir(root.parent)
        prep_created = False
        phase_hook("after_promote_before_snapshot")
        immutable_after_promote = immutable_snapshot()
        output = root / "authority_receipt.json"
        root_tree = exact_tree(root)
        if (immutable_after_promote != immutable_pre
                or root_tree["inventory"] != [["authority_receipt.json", sha(output), output.stat().st_size]]
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
            "authority_registration_tree": root_tree}


def synthetic_self_test() -> int:
    cases = {}
    with tempfile.TemporaryDirectory(prefix="v490-authority-publish-") as directory:
        base = Path(directory)

        def make_case(name):
            case = base / name
            case.mkdir()
            guard = case / "guard.bin"
            with guard.open("xb") as stream:
                stream.write(b"frozen\n")
                stream.flush()
                os.fsync(stream.fileno())
            fsync_dir(case)
            snap = lambda: {"guard": {"sha256": sha(guard), "logical_bytes": guard.stat().st_size}}
            return case / "authority", case / "authority.registration-prep", guard, snap

        root, prep, guard, snap = make_case("normal")
        normal = publish_exact1(root, prep, {"passed": True}, snap)
        cases["normal"] = normal["committed_success"] is True and not prep.exists()

        root, prep, guard, snap = make_case("precommit-signal")
        try:
            publish_exact1(root, prep, {"passed": True}, snap,
                           lambda phase: os.kill(os.getpid(), signal.SIGTERM)
                           if phase == "owned_prep_unblocked" else None)
            cases["precommit_signal_cleanup"] = False
        except ControlledSignal:
            cases["precommit_signal_cleanup"] = not root.exists() and not prep.exists()

        root, prep, guard, snap = make_case("deferred-signal")
        if hasattr(signal, "pthread_sigmask"):
            deferred = publish_exact1(root, prep, {"passed": True}, snap,
                                      lambda phase: os.kill(os.getpid(), signal.SIGTERM)
                                      if phase == "commit_window_blocked" else None)
            cases["deferred_signal"] = deferred["deferred_signal"] == signal.SIGTERM
        else:
            cases["deferred_signal"] = os.name == "nt"

        root, prep, guard, snap = make_case("postpromote-drift")
        def drift(phase):
            if phase == "after_promote_before_snapshot":
                with guard.open("ab") as stream:
                    stream.write(b"drift\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        try:
            publish_exact1(root, prep, {"passed": True}, snap, drift)
            cases["postpromote_drift"] = False
        except RuntimeError as error:
            cases["postpromote_drift"] = root.exists() and "post-promote" in str(error)
    result = {"passed": all(cases.values()), "checks": cases}
    result["checks_sha256"] = csha(cases)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


def absence_map(value: dict, expected_keys: set[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("absence key set")
    result = {}
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != {"path"}:
            raise RuntimeError(f"absence schema: {name}")
        path = Path(row["path"])
        if path != path.resolve():
            raise RuntimeError(f"absence path: {name}")
        result[name] = str(path)
    return result


def snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, str]) -> dict:
    file_rows = []
    for name, target in sorted(files.items()):
        path = Path(target)
        if path != path.resolve() or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"snapshot file: {name}")
        file_rows.append({"name": name, "path": str(path), "sha256": sha(path),
                          "logical_bytes": path.stat().st_size})
    tree_rows = {name: exact_tree(target) for name, target in sorted(trees.items())}
    absence_rows = []
    for name, target in sorted(absences.items()):
        if os.path.lexists(target):
            raise RuntimeError(f"snapshot absence: {name}")
        absence_rows.append({"name": name, "path": target, "absent": True})
    body = {"files": file_rows, "trees": tree_rows, "absences": absence_rows}
    return {**body, "snapshot_sha256": csha(body)}


def live_processes(fragments: set[str]) -> list[dict]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(fragment in command for fragment in fragments):
            found.append({"pid": int(proc.name), "cmdline": command})
    return found


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--static-receipt", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1632
            or pending(contract)):
        raise RuntimeError("contract frozen schema")
    authority_schema = contract["authority_receipt_contract"]
    if (set(authority_schema) != AUTHORITY_SCHEMA_KEYS
            or authority_schema.get("authorization_exact") != AUTHORIZATION
            or authority_schema.get("runtime_observation_exact") != RUNTIME
            or contract.get("execution_boundary") != EXECUTION_BOUNDARY):
        raise RuntimeError("contract authority boundary")

    materializer_record = regular(args.materializer_source, args.materializer_sha,
                                  args.materializer_source.stat().st_size)
    if (args.materializer_source.resolve() != Path(__file__).resolve()
            or materializer_record != contract["authority_materializer_source"]):
        raise RuntimeError("materializer binding")
    if set(contract["source_closure"]) != SOURCE_ROLES:
        raise RuntimeError("source role set")
    sources = {name: record(value) for name, value in contract["source_closure"].items()}
    if contract["source_closure_sha256"] != csha(contract["source_closure"]):
        raise RuntimeError("source closure digest")
    static_record = record(contract["static_receipt"])
    if args.static_receipt.resolve() != Path(static_record["path"]):
        raise RuntimeError("static receipt path")

    formal = json.loads(Path(sources["repair_preregistration"]["path"]).read_text())
    static = json.loads(args.static_receipt.read_text())
    formal_schema = contract["repair_formal_schema"]
    if (set(formal_schema) != FORMAL_SCHEMA_KEYS
            or set(formal) != set(formal_schema["top_keys"])
            or formal.get("format") != formal_schema["format"]
            or formal.get("status") != formal_schema["status"]
            or formal.get("authorization") != formal_schema["authorization_exact"]
            or formal.get("runtime_observation") != formal_schema["runtime_observation_exact"]):
        raise RuntimeError("repair formal schema")
    static_schema = contract["static_receipt_contract"]
    if (set(static_schema) != STATIC_SCHEMA_KEYS
            or set(static) != STATIC_TOP_KEYS or set(static) != set(static_schema["top_keys"])
            or static.get("format") != static_schema["format"]
            or static.get("status") != static_schema["status"] or static.get("passed") is not True):
        raise RuntimeError("static receipt schema")
    if (static_schema["check_keys"] != STATIC_CHECK_KEYS
            or static_schema["check_key_set_sha256"] != STATIC_KEYSET_SHA
            or static_schema["checks_sha256"] != STATIC_CHECKS_SHA
            or static.get("checks") != {key: True for key in STATIC_CHECK_KEYS}
            or static.get("check_keys") != STATIC_CHECK_KEYS
            or static.get("check_key_set_sha256") != STATIC_KEYSET_SHA
            or static.get("checks_sha256") != STATIC_CHECKS_SHA):
        raise RuntimeError("static checks")
    if (static_schema.get("runtime_observation_exact") != STATIC_RUNTIME
            or static.get("runtime_observation") != STATIC_RUNTIME
            or any(static.get(key) is not False for key in
                   ("readonly_reconciliation_authorized", "training_authorized", "submission_authorized"))):
        raise RuntimeError("static authority boundary")
    if (static.get("repair_preregistration") != sources["repair_preregistration"]
            or static.get("design_contract") != sources["repair_design_contract"]
            or static.get("materializer_source") != sources["repair_formal_materializer"]
            or static.get("reconciler_r2_source") != sources["reconciler_r2"]
            or static.get("static_auditor_source") != sources["static_auditor"]
            or static.get("source_closure") != formal["source_closure"]
            or static.get("source_closure_sha256") != csha(formal["source_closure"])):
        raise RuntimeError("static aliases")
    failure = contract["static_failure_ancestry"]
    if set(failure) != FAILURE_ANCESTRY_KEYS:
        raise RuntimeError("static failure ancestry schema")
    for key in sorted(FAILURE_ANCESTRY_KEYS):
        if static.get(key) != failure[key]:
            raise RuntimeError(f"static failure ancestry: {key}")
    record(failure["superseded_static_source"])
    record(failure["failed_static_process_receipt"])
    record(failure["v489_failed_static_source"])
    record(failure["v489_failed_static_process_receipt"])
    if (exact_tree(Path(failure["failed_static_process_receipt"]["path"]).parent)
            != failure["failed_static_execution_tree"]
            or exact_tree(Path(failure["v489_failed_static_process_receipt"]["path"]).parent)
            != failure["v489_failed_static_execution_tree"]):
        raise RuntimeError("failed static evidence current tree")
    failed_f88 = json.loads(Path(failure["failed_static_process_receipt"]["path"]).read_text())
    failed_v489 = json.loads(Path(failure["v489_failed_static_process_receipt"]["path"]).read_text())
    for failed in (failed_f88, failed_v489):
        if (failed.get("status") != "failed_no_retry" or failed.get("passed") is not False
                or failed.get("retry_authorized") is not False
                or failed.get("static_invocations") != 1 or failed.get("r2_invocations") != 0
                or failed.get("wrapper_invocations") != 0
                or failed.get("cleanup", {}).get("group_empty") is not True):
            raise RuntimeError("failed static process semantics")
    if failed_v489.get("prior_predeploy_transport_failure") != failure["v489_predeploy_transport_failure"]:
        raise RuntimeError("v489 transport disclosure")

    repair_tree = verify_tree(contract["repair_formal_registration_tree"])
    static_tree = verify_tree(contract["postregistration_static_registration_tree"])
    execution_tree = verify_tree(contract["static_execution_evidence_tree"])
    f813_tree = verify_tree(contract["f813_registration_tree"])
    failed_tree = verify_tree(contract["failed_attempt_tree"])
    evidence_tree = verify_tree(contract["formal_materialization_evidence_tree"])
    if (repair_tree["inventory"] != [["preregistration.json", sources["repair_preregistration"]["sha256"],
                                      sources["repair_preregistration"]["logical_bytes"]]]
            or static_tree["inventory"] != [["static_audit.json", static_record["sha256"],
                                              static_record["logical_bytes"]]]):
        raise RuntimeError("exact1 registration trees")
    process_record = next((row for row in execution_tree["inventory"]
                           if row[0] == "process_receipt.json"), None)
    if process_record is None:
        raise RuntimeError("static process evidence")
    process_path = Path(contract["static_execution_evidence_tree"]["root"], process_record[0])
    process = json.loads(process_path.read_text())
    if (process.get("passed") is not True or process.get("returncode") != 0
            or process.get("static_invocations") != 1 or process.get("r2_invocations") != 0
            or process.get("wrapper_invocations") != 0):
        raise RuntimeError("static execution semantics")
    if (static.get("f813_registration_tree") != f813_tree
            or static.get("failed_reconciliation_attempt_tree") != failed_tree):
        raise RuntimeError("static tree aliases")

    lineage = contract["lineage"]
    if set(lineage) != {"authority_root", "attempt_root", "attempt_prep",
                       "transparent_static_receipt_path", "qualification_root",
                       "postregistration_static_root"}:
        raise RuntimeError("lineage schema")
    if lineage != {"authority_root": str(AUTHORITY_ROOT), "attempt_root": str(ATTEMPT_ROOT),
                   "attempt_prep": str(ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + ".attempt-prep")),
                   "transparent_static_receipt_path": str(TRANSPARENT_PATH),
                   "qualification_root": str(QUALIFICATION_ROOT),
                   "postregistration_static_root": str(STATIC_ROOT)}:
        raise RuntimeError("lineage values")
    if Path(contract["postregistration_static_registration_tree"]["root"]) != STATIC_ROOT:
        raise RuntimeError("static tree root lineage")
    root = args.authority_root
    if (root != root.resolve() or str(root) != lineage["authority_root"]
            or not root.parent.is_dir() or root.parent.is_symlink()):
        raise RuntimeError("authority root")
    prep = root.with_name(root.name + ".registration-prep")
    historical_keys = {"superseded_static_root", "superseded_static_prep",
                       "v489_superseded_static_root", "v489_superseded_static_prep",
                       "fresh_static_prep", "authority_root", "authority_prep",
                       "attempt_root", "attempt_prep", "transparent_output",
                       "transparent_tmp", "qualification_root"}
    current_keys = historical_keys - {"authority_root"}
    historical = absence_map(contract["historical_absences"], historical_keys)
    current = absence_map(contract["current_absences_after_authority"], current_keys)
    if (historical["authority_root"] != str(root) or historical["authority_prep"] != str(prep)
            or {key: historical[key] for key in current_keys} != current):
        raise RuntimeError("absence crossbinding")
    expected_f88_absences = {key: {"path": historical[key], "absent": True}
                             for key in ("superseded_static_root", "superseded_static_prep")}
    expected_v489_absences = {key: {"path": historical[key], "absent": True}
                              for key in ("v489_superseded_static_root", "v489_superseded_static_prep")}
    if (failure["superseded_static_absences"] != expected_f88_absences
            or failure["v489_superseded_static_absences"] != expected_v489_absences):
        raise RuntimeError("failed static absence ancestry")
    if any(os.path.lexists(target) for target in historical.values()):
        raise RuntimeError("historical absence drift")
    static_absences = static.get("required_absences")
    if (set(static_absences or {}) != historical_keys
            or any(set(row) != {"path", "absent"} or row["absent"] is not True
                   or row["path"] != historical[name] for name, row in static_absences.items())):
        raise RuntimeError("static absence aliases")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source),
             "static_receipt": static_record["path"],
             "failure::f88_source": failure["superseded_static_source"]["path"],
             "failure::f88_process_receipt": failure["failed_static_process_receipt"]["path"],
             "failure::v489_source": failure["v489_failed_static_source"]["path"],
             "failure::v489_process_receipt": failure["v489_failed_static_process_receipt"]["path"]}
    for name, value in sources.items():
        files[f"source::{name}"] = value["path"]
    trees = {"repair_formal_REG": contract["repair_formal_registration_tree"]["root"],
             "postregistration_static_REG": contract["postregistration_static_registration_tree"]["root"],
             "static_execution_evidence": contract["static_execution_evidence_tree"]["root"],
             "f813_REG": contract["f813_registration_tree"]["root"],
             "failed_attempt": contract["failed_attempt_tree"]["root"],
             "formal_evidence": contract["formal_materialization_evidence_tree"]["root"],
             "f88_failed_static": str(Path(failure["failed_static_process_receipt"]["path"]).parent),
             "v489_failed_static": str(Path(failure["v489_failed_static_process_receipt"]["path"]).parent)}
    pre = snapshot(files, trees, historical)
    post = snapshot(files, trees, historical)
    if pre != post:
        raise RuntimeError("input drift")
    live = live_processes({sources["reconciler_r2"]["path"], sources["execution_wrapper"]["path"]})
    if live:
        raise RuntimeError("live reconciler or wrapper")
    terminal = json.loads(Path(contract["failed_attempt_tree"]["root"], "terminal_receipt.json").read_text())
    if (terminal.get("status") != "failed_no_retry" or terminal.get("passed") is not False
            or terminal.get("retry_authorized") is not False):
        raise RuntimeError("old v487 retry boundary")

    checks = {key: True for key in AUTH_CHECK_KEYS}
    receipt = {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": materializer_record,
        "repair_preregistration": sources["repair_preregistration"],
        "repair_design_contract": sources["repair_design_contract"],
        "repair_formal_materializer": sources["repair_formal_materializer"],
        "reconciler_r2_source": sources["reconciler_r2"],
        "static_auditor_source": sources["static_auditor"],
        "execution_wrapper_source": sources["execution_wrapper"],
        "static_receipt": static_record,
        "repair_formal_registration_tree": repair_tree,
        "postregistration_static_registration_tree": static_tree,
        "static_execution_evidence_tree": execution_tree,
        "f813_registration_tree": f813_tree,
        "failed_attempt_tree": failed_tree,
        "formal_materialization_evidence_tree": evidence_tree,
        "attempt_root": lineage["attempt_root"],
        "transparent_static_receipt_path": lineage["transparent_static_receipt_path"],
        "historical_absences": {name: {"path": target, "absent": True}
                                for name, target in sorted(historical.items())},
        "required_absences": {name: {"path": target, "absent": True}
                              for name, target in sorted(current.items())},
        "checks": checks, "checks_sha256": csha(checks),
        "input_pre_snapshot": pre, "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True,
        "authorization": contract["authority_receipt_contract"]["authorization_exact"],
        "runtime_observation": contract["authority_receipt_contract"]["runtime_observation_exact"],
        **failure,
    }
    if (set(receipt) != AUTH_TOP_KEYS or set(receipt) != set(authority_schema["top_keys"])
            or authority_schema["format"] != OUTPUT_FORMAT
            or authority_schema["status"] != OUTPUT_STATUS
            or authority_schema["check_keys"] != AUTH_CHECK_KEYS
            or authority_schema["check_key_set_sha256"] != AUTH_KEYSET_SHA
            or authority_schema["checks_sha256"] != AUTH_CHECKS_SHA
            or csha(AUTH_CHECK_KEYS) != AUTH_KEYSET_SHA or csha(checks) != AUTH_CHECKS_SHA):
        raise RuntimeError("authority receipt schema")

    stable_absences = {name: target for name, target in current.items() if target != str(prep)}
    immutable_snapshot = lambda: snapshot(files, trees, stable_absences)
    publication = publish_exact1(root, prep, receipt, immutable_snapshot)
    output = root / "authority_receipt.json"
    print(json.dumps({"path": str(output), "sha256": sha(output),
                      "logical_bytes": output.stat().st_size,
                      "authority_registration_tree": publication["authority_registration_tree"],
                      "committed_success": publication["committed_success"],
                      "deferred_signal": publication["deferred_signal"],
                      "reconciler_r2_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
