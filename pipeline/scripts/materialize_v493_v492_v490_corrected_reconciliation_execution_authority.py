#!/usr/bin/env python3
"""Atomically register the v493 corrected-inner authority; run no execution source."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CONTRACT_PATH = ROOT / "pipeline/scripts/v493_v492_v490_corrected_reconciliation_execution_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v493-v492-v490-corrected-reconciliation-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v493-v492-v490-corrected-reconciliation-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_v493_corrected_reconciliation_outer_attempt"
AUTHORITY_ROOT = J / "v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825"
OUTER_EVIDENCE_ROOT = J / "v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825"
INNER_ATTEMPT_ROOT = J / "v493_v492_v490_corrected_reconciliation_inner_attempt_seed1635_20260825"
FAILED_V492_OUTER_ROOT = J / "v492_v491_v490_reconciliation_outer_execution_evidence_seed1634_20260825"
V492_AUTHORITY_ROOT = J / "v492_v491_v490_reconciliation_execution_authority_seed1634_20260825"
OLD_V490_INNER_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SOURCE_ROLES = {
    "v492_authority_contract", "v492_authority_materializer", "v492_outer_wrapper",
    "superseded_v490_inner_wrapper", "corrected_inner_wrapper", "reconciler_r2",
    "outer_execution_wrapper", "phase_a_design_contract",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_closure_sha256",
    "authority_materializer_source", "v492_authority_contract", "v492_authority_receipt",
    "v492_authority_registration_tree", "v492_authority_materialization_evidence_tree",
    "v492_authority_materializer_process_receipt", "v492_outer_failure_tree",
    "v492_outer_terminal_receipt", "v492_outer_failure_ancestry",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "phase_a_design_contract_record", "historical_absences",
    "current_absences_after_authority", "authority_receipt_contract", "execution_boundary",
}
AUTH_TOP_KEYS = {
    "format", "status", "passed", "authority_design_contract",
    "authority_materializer_source", "outer_execution_wrapper_source",
    "corrected_inner_wrapper_source", "superseded_v490_inner_wrapper_source",
    "phase_a_design_contract_source", "reconciler_r2_source",
    "v492_authority_contract", "v492_authority_materializer_source",
    "v492_outer_wrapper_source", "v492_authority_receipt",
    "v492_authority_registration_tree", "v492_authority_materialization_evidence_tree",
    "v492_authority_materializer_process_receipt", "failed_v492_outer_execution_tree",
    "failed_v492_outer_terminal_receipt",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "source_closure", "source_closure_sha256",
    "outer_evidence_root", "inner_attempt_root", "transparent_static_receipt_path",
    "historical_absences", "required_absences", "checks", "check_keys",
    "check_key_set_sha256", "checks_sha256", "input_pre_snapshot",
    "input_post_snapshot", "input_snapshots_exactly_equal", "authorization",
    "runtime_observation", "execution_boundary",
}
AUTH_CHECK_KEYS = sorted({
    "authority_contract_current", "authority_materializer_current",
    "corrected_inner_current", "corrected_inner_diff_exact", "current_absences",
    "execution_boundary", "f813_exact2", "failed_v492_outer_exact4",
    "failed_v492_outer_no_retry_partition", "failed_v492_outer_terminal_exact",
    "gpu_empty", "historical_absences", "input_snapshots_equal", "no_live_process",
    "outer_wrapper_current", "phase_a_contract_current", "phase_a_contract_spec_exact2",
    "postregistration_static_exact1", "r2_current", "repair_formal_exact1",
    "source_closure_current", "v492_authority_contract_current", "v492_authority_exact3",
    "v492_authority_materialization_evidence_exact6",
    "v492_authority_materializer_process_exact", "v492_outer_wrapper_current",
})
AUTHORIZATION = {
    "outer_execution_wrapper_authorized": True,
    "outer_attempts_authorized": 1,
    "outer_attempts_consumed": 0,
    "retry_authorized": False,
    "direct_corrected_inner_authorized": False,
    "direct_r2_authorized": False,
    "nested_corrected_inner_invocations_authorized": 1,
    "nested_r2_invocations_authorized": 1,
    "nested_corrected_inner_only_via_outer": True,
    "nested_r2_only_via_corrected_inner": True,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "folds_authorized": 0,
    "policy_updates": 0,
    "s1_authorized": False,
    "zero_update_authorized": False,
    "rl_authorized": False,
    "submission_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME = {
    "execution_authority_materialized": True,
    "outer_execution_wrapper_executed": False,
    "corrected_inner_wrapper_executed": False,
    "reconciler_r2_executed": False,
    "transparent_receipt_created": False,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True,
    "outer_execution_wrapper_invocations": 0,
    "corrected_inner_wrapper_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_invocations": 0,
    "training_invocations": 0,
    "reward_reads": 0,
    "dev_hidden_final_outcome_reads": 0,
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


def regular(path: Path | str, digest: str | None = None, logical_bytes: int | None = None) -> dict:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"file closure: {path}")
    actual = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if digest is not None and actual["sha256"] != digest:
        raise RuntimeError(f"file digest: {path}")
    if logical_bytes is not None and actual["logical_bytes"] != logical_bytes:
        raise RuntimeError(f"file bytes: {path}")
    return actual


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
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def rooted_tree(root: Path | str) -> dict:
    return {"root": str(Path(root)), **exact_tree(root)}


def verify_tree(spec: dict) -> dict:
    keys = {"root", "inventory", "file_count", "logical_file_bytes",
            "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise RuntimeError("tree schema")
    actual = rooted_tree(spec["root"])
    if actual != spec:
        raise RuntimeError(f"tree closure: {spec['root']}")
    return actual


def decode_absences(value: dict, expected_keys: set[str]) -> dict[str, Path]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("absence key set")
    result = {}
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != {"path"} or not isinstance(row["path"], str):
            raise RuntimeError(f"absence row: {name}")
        path = Path(row["path"])
        if path != path.resolve():
            raise RuntimeError(f"absence path: {name}")
        result[name] = path
    return result


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def copy_fsync(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
        dst.flush()
        os.fsync(dst.fileno())


def directory_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("directory ownership")
    return stat.st_dev, stat.st_ino


def cleanup_owned(path: Path, identity: tuple[int, int]) -> None:
    if path.exists() and not path.is_symlink() and directory_identity(path) == identity:
        shutil.rmtree(path)
        fsync_dir(path.parent)


def live_processes(fragments: set[str]) -> list[dict]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(fragment in cmd for fragment in fragments):
            found.append({"pid": int(proc.name), "cmdline": cmd})
    return found


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("gpu query")
    return [line for line in result.stdout.splitlines() if line.strip()]


def snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, Path]) -> dict:
    file_rows = []
    for name, target in sorted(files.items()):
        row = regular(target)
        file_rows.append({"name": name, **row})
    tree_rows = {name: rooted_tree(target) for name, target in sorted(trees.items())}
    absence_rows = []
    for name, target in sorted(absences.items()):
        absence_rows.append({"name": name, "path": str(target), "absent": not os.path.lexists(target)})
    if not all(row["absent"] for row in absence_rows):
        raise RuntimeError("snapshot absence")
    value = {"files": file_rows, "trees": tree_rows, "absences": absence_rows}
    value["snapshot_sha256"] = csha(value)
    return value


def validate_corrected_inner(path: Path) -> None:
    """Independently establish the only permitted semantic delta from frozen 733."""
    text = path.read_text()
    parsed = ast.parse(text)
    main_node = next((node for node in parsed.body
                      if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    popens = [] if main_node is None else [
        node for node in ast.walk(main_node) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"]
    normalized = text.replace(" ", "")
    if (len(popens) != 1
            or 'phase_contract_spec["logical_bytes"]' in text
            or "phase_contract_spec['logical_bytes']" in text
            or "PHASE_A_DESIGN_CONTRACT_BYTES=43960" not in normalized
            or 'set(spec)!={"path","sha256"}' not in normalized
            or "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64" not in text
            or "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377" not in text):
        raise RuntimeError("corrected inner phase-contract repair")
    imported = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    if imported & {"torch", "transformers", "rlinf"}:
        raise RuntimeError("corrected inner forbidden import")


def synthetic_self_test() -> int:
    checks = {
        "authority_top_count": len(AUTH_TOP_KEYS) == 39,
        "check_count": len(AUTH_CHECK_KEYS) == 26,
        "source_roles": len(SOURCE_ROLES) == 8,
        "authorization_boundary": (
            AUTHORIZATION["outer_execution_wrapper_authorized"] is True
            and AUTHORIZATION["direct_corrected_inner_authorized"] is False
            and AUTHORIZATION["direct_r2_authorized"] is False
            and AUTHORIZATION["retry_authorized"] is False
        ),
        "fresh_roots": all("v493_" in path.name for path in
                           (AUTHORITY_ROOT, OUTER_EVIDENCE_ROOT, INNER_ATTEMPT_ROOT)),
    }
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        good = {"a": {"path": str(base / "a")}, "b": {"path": str(base / "b")}}
        checks["absence_decode"] = set(decode_absences(good, {"a", "b"})) == {"a", "b"}
        try:
            decode_absences({"a": str(base / "a")}, {"a"})
            checks["absence_tamper"] = False
        except RuntimeError:
            checks["absence_tamper"] = True
    checks["main_pending_resolves_global"] = (
        "pending" not in main.__code__.co_varnames
        and pending({"actual_contract_shape": True}) is False
        and pending({"marker": "PENDING"}) is True
    )
    print(json.dumps({"passed": all(checks.values()), "checks": checks,
                      "checks_sha256": csha(checks)}, sort_keys=True))
    return 0 if all(checks.values()) else 3


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1635
            or pending(contract)):
        raise RuntimeError("contract frozen schema")
    materializer_record = regular(args.materializer_source, args.materializer_sha,
                                  args.materializer_source.stat().st_size)
    if (args.materializer_source.resolve() != Path(__file__).resolve()
            or materializer_record != contract["authority_materializer_source"]):
        raise RuntimeError("materializer binding")
    if set(contract["source_closure"]) != SOURCE_ROLES:
        raise RuntimeError("source roles")
    sources = {name: record(value) for name, value in contract["source_closure"].items()}
    if sources != contract["source_closure"] or csha(sources) != contract["source_closure_sha256"]:
        raise RuntimeError("source closure")
    phase_record = sources["phase_a_design_contract"]
    if (phase_record != contract["phase_a_design_contract_record"]
            or phase_record["sha256"] != "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
            or phase_record["logical_bytes"] != 43960):
        raise RuntimeError("phase contract current record")
    validate_corrected_inner(Path(sources["corrected_inner_wrapper"]["path"]))

    repair_tree = verify_tree(contract["repair_formal_registration_tree"])
    static_tree = verify_tree(contract["postregistration_static_registration_tree"])
    f813_tree = verify_tree(contract["f813_registration_tree"])
    if repair_tree["file_count"] != 1 or static_tree["file_count"] != 1 or f813_tree["file_count"] != 2:
        raise RuntimeError("frozen tree cardinality")
    f813_path = Path(f813_tree["root"]) / "preregistration.json"
    f813 = json.loads(f813_path.read_text())
    phase_spec = f813.get("frozen_parent", {}).get("phase_a_design_contract")
    if (not isinstance(phase_spec, dict) or set(phase_spec) != {"path", "sha256"}
            or phase_spec != {"path": phase_record["path"], "sha256": phase_record["sha256"]}):
        raise RuntimeError("F813 phase contract exact2 projection")

    v492_contract = json.loads(Path(sources["v492_authority_contract"]["path"]).read_text())
    if record(contract["v492_authority_contract"]) != sources["v492_authority_contract"]:
        raise RuntimeError("v492 contract alias")
    v492_receipt_record = record(contract["v492_authority_receipt"])
    v492_receipt = json.loads(Path(v492_receipt_record["path"]).read_text())
    old_schema = v492_contract.get("authority_receipt_contract", {})
    if (set(v492_receipt) != set(old_schema.get("top_keys", []))
            or len(v492_receipt) != 44 or len(v492_receipt.get("checks", {})) != 33
            or v492_receipt.get("format") != old_schema.get("format")
            or v492_receipt.get("status") != old_schema.get("status")
            or v492_receipt.get("passed") is not True
            or v492_receipt.get("checks") != {key: True for key in old_schema.get("check_keys", [])}
            or v492_receipt.get("checks_sha256") != old_schema.get("checks_sha256")
            or v492_receipt.get("authorization") != old_schema.get("authorization_exact")
            or v492_receipt.get("runtime_observation") != old_schema.get("runtime_observation_exact")):
        raise RuntimeError("v492 authority receipt")
    v492_auth_tree = verify_tree(contract["v492_authority_registration_tree"])
    if (v492_auth_tree["file_count"] != 3
            or not any(row == ["authority_receipt.json", v492_receipt_record["sha256"],
                               v492_receipt_record["logical_bytes"]]
                       for row in v492_auth_tree["inventory"])):
        raise RuntimeError("v492 authority exact3")
    v492_evidence_tree = verify_tree(contract["v492_authority_materialization_evidence_tree"])
    v492_process_record = record(contract["v492_authority_materializer_process_receipt"])
    v492_process = json.loads(Path(v492_process_record["path"]).read_text())
    if (v492_evidence_tree["file_count"] != 6
            or not any(row == ["process_receipt.json", v492_process_record["sha256"],
                               v492_process_record["logical_bytes"]]
                       for row in v492_evidence_tree["inventory"])
            or v492_process.get("passed") is not True
            or v492_process.get("materializer_invocations") != 1
            or v492_process.get("outer_wrapper_invocations") != 0
            or v492_process.get("inner_wrapper_invocations") != 0
            or v492_process.get("r2_invocations") != 0
            or v492_process.get("retry_authorized") is not False
            or v492_process.get("pre_post_snapshots_exactly_equal") is not True
            or v492_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v492 authority materialization evidence")

    failed_tree = verify_tree(contract["v492_outer_failure_tree"])
    failed_terminal_record = record(contract["v492_outer_terminal_receipt"])
    failed_terminal = json.loads(Path(failed_terminal_record["path"]).read_text())
    stderr_path = Path(failed_tree["root"]) / "v490_wrapper_stderr.log"
    failure_spec = contract["v492_outer_failure_ancestry"]
    expected_failure_spec = {
        "failed_status": "failed_no_retry",
        "superseded_outer_wrapper_invocations": 1,
        "superseded_inner_wrapper_invocations": 1,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "root_cause": "F813 phase_a_design_contract exact2 lacks logical_bytes but superseded 733 indexed it",
        "corrected_rule": "rehash actual phase contract to full record then compare F813 exact2 projection",
    }
    if (failed_tree["file_count"] != 4 or failure_spec != expected_failure_spec
            or not any(row == ["terminal_receipt.json", failed_terminal_record["sha256"],
                               failed_terminal_record["logical_bytes"]]
                       for row in failed_tree["inventory"])
            or failed_terminal.get("status") != "failed_no_retry"
            or failed_terminal.get("passed") is not False
            or failed_terminal.get("nested_v490_wrapper_invocations") != 1
            or failed_terminal.get("retry_authorized") is not False
            or failed_terminal.get("cleanup", {}).get("reaped") is not True
            or failed_terminal.get("cleanup", {}).get("group_empty") is not True
            or failed_terminal.get("transparent_present") is not False
            or "KeyError" not in stderr_path.read_text()
            or "phase_contract_spec[\"logical_bytes\"]" not in stderr_path.read_text()):
        raise RuntimeError("v492 outer failed-no-retry ancestry")

    lineage = contract["lineage"]
    expected_lineage = {
        "execution_authority_root": str(AUTHORITY_ROOT),
        "authority_prep": str(AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")),
        "outer_evidence_root": str(OUTER_EVIDENCE_ROOT),
        "outer_evidence_prep": str(OUTER_EVIDENCE_ROOT.with_name(OUTER_EVIDENCE_ROOT.name + ".outer-prep")),
        "inner_attempt_root": str(INNER_ATTEMPT_ROOT),
        "inner_attempt_prep": str(INNER_ATTEMPT_ROOT.with_name(INNER_ATTEMPT_ROOT.name + ".attempt-prep")),
        "transparent_static_receipt_path": str(TRANSPARENT_PATH),
        "qualification_root": str(QUALIFICATION_ROOT),
    }
    if lineage != expected_lineage:
        raise RuntimeError("lineage")
    root = args.authority_root
    prep = root.with_name(root.name + ".registration-prep")
    if root != AUTHORITY_ROOT or root.resolve() != root or root.parent.is_symlink() or not root.parent.is_dir():
        raise RuntimeError("authority root")
    historical_keys = {
        "superseded_static_root", "superseded_static_prep", "v489_superseded_static_root",
        "v489_superseded_static_prep", "fresh_static_prep", "v490_authority_prep",
        "old_v490_inner_attempt_root", "old_v490_inner_attempt_prep", "transparent_output",
        "transparent_tmp", "qualification_root", "superseded_v491_authority_root",
        "superseded_v491_authority_prep", "superseded_v491_outer_evidence_root",
        "superseded_v491_outer_evidence_prep", "v492_authority_prep",
        "failed_v492_outer_evidence_prep", "execution_authority_root",
        "execution_authority_prep", "outer_evidence_root", "outer_evidence_prep",
        "corrected_inner_attempt_root", "corrected_inner_attempt_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = decode_absences(contract["historical_absences"], historical_keys)
    current = decode_absences(contract["current_absences_after_authority"], current_keys)
    if (historical["execution_authority_root"] != root
            or historical["execution_authority_prep"] != prep
            or historical["failed_v492_outer_evidence_prep"] != FAILED_V492_OUTER_ROOT.with_name(FAILED_V492_OUTER_ROOT.name + ".outer-prep")
            or historical["old_v490_inner_attempt_root"] != OLD_V490_INNER_ROOT
            or {key: historical[key] for key in current_keys} != current
            or any(os.path.lexists(path) for path in historical.values())):
        raise RuntimeError("absence prestate")
    live_fragments = {row["path"] for role, row in sources.items()
                      if role in {"corrected_inner_wrapper", "outer_execution_wrapper", "reconciler_r2"}}
    if live_processes(live_fragments) or gpu_processes():
        raise RuntimeError("process or GPU prestate")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source),
             "v492_authority_receipt": v492_receipt_record["path"],
             "v492_process_receipt": v492_process_record["path"],
             "failed_v492_terminal": failed_terminal_record["path"]}
    files.update({f"source::{name}": row["path"] for name, row in sources.items()})
    trees = {
        "v492_authority_REG": contract["v492_authority_registration_tree"]["root"],
        "v492_authority_materialization": contract["v492_authority_materialization_evidence_tree"]["root"],
        "failed_v492_outer": contract["v492_outer_failure_tree"]["root"],
        "repair_formal_REG": contract["repair_formal_registration_tree"]["root"],
        "postregistration_static_REG": contract["postregistration_static_registration_tree"]["root"],
        "f813_REG": contract["f813_registration_tree"]["root"],
    }
    historical_pre = snapshot(files, trees, historical)
    pre = snapshot(files, trees, current)
    stable_absences = {name: path for name, path in current.items()
                       if name != "execution_authority_prep"}
    stable_pre = snapshot(files, trees, stable_absences)
    post = snapshot(files, trees, current)
    if (pre != post or historical_pre["files"] != pre["files"]
            or historical_pre["trees"] != pre["trees"]):
        raise RuntimeError("input drift")

    schema = contract["authority_receipt_contract"]
    checks = {key: True for key in AUTH_CHECK_KEYS}
    receipt = {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": materializer_record,
        "corrected_inner_wrapper_source": sources["corrected_inner_wrapper"],
        "outer_execution_wrapper_source": sources["outer_execution_wrapper"],
        "superseded_v490_inner_wrapper_source": sources["superseded_v490_inner_wrapper"],
        "phase_a_design_contract_source": phase_record,
        "reconciler_r2_source": sources["reconciler_r2"],
        "v492_authority_contract": sources["v492_authority_contract"],
        "v492_authority_materializer_source": sources["v492_authority_materializer"],
        "v492_outer_wrapper_source": sources["v492_outer_wrapper"],
        "v492_authority_receipt": v492_receipt_record,
        "v492_authority_registration_tree": v492_auth_tree,
        "v492_authority_materialization_evidence_tree": v492_evidence_tree,
        "v492_authority_materializer_process_receipt": v492_process_record,
        "failed_v492_outer_execution_tree": failed_tree,
        "failed_v492_outer_terminal_receipt": failed_terminal_record,
        "repair_formal_registration_tree": repair_tree,
        "postregistration_static_registration_tree": static_tree,
        "f813_registration_tree": f813_tree,
        "source_closure": sources, "source_closure_sha256": csha(sources),
        "outer_evidence_root": str(OUTER_EVIDENCE_ROOT),
        "inner_attempt_root": str(INNER_ATTEMPT_ROOT),
        "transparent_static_receipt_path": str(TRANSPARENT_PATH),
        "historical_absences": {name: {"path": str(path), "absent": True}
                                for name, path in sorted(historical.items())},
        "required_absences": {name: {"path": str(path), "absent": True}
                              for name, path in sorted(current.items())},
        "checks": checks, "check_keys": AUTH_CHECK_KEYS,
        "check_key_set_sha256": csha(AUTH_CHECK_KEYS), "checks_sha256": csha(checks),
        "input_pre_snapshot": pre, "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True,
        "authorization": AUTHORIZATION, "runtime_observation": RUNTIME,
        "execution_boundary": EXECUTION_BOUNDARY,
    }
    if (set(schema) != {"format", "status", "top_keys", "check_keys",
                        "check_key_set_sha256", "checks_sha256", "authorization_exact",
                        "runtime_observation_exact"}
            or set(receipt) != AUTH_TOP_KEYS or set(receipt) != set(schema["top_keys"])
            or schema["format"] != OUTPUT_FORMAT or schema["status"] != OUTPUT_STATUS
            or schema["check_keys"] != AUTH_CHECK_KEYS
            or schema["check_key_set_sha256"] != csha(AUTH_CHECK_KEYS)
            or schema["checks_sha256"] != csha(checks)
            or schema["authorization_exact"] != AUTHORIZATION
            or schema["runtime_observation_exact"] != RUNTIME
            or contract["execution_boundary"] != EXECUTION_BOUNDARY):
        raise RuntimeError("authority receipt schema")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    pending_signal = {"value": None}
    committed = False
    identity = None
    old_mask = None
    def on_signal(signum, _frame):
        pending_signal["value"] = signum
        if not committed:
            raise ControlledSignal(signum)
    for sig in old_handlers:
        signal.signal(sig, on_signal)
    try:
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        prep.mkdir()
        identity = directory_identity(prep)
        fsync_dir(prep)
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        atomic_json(prep / "authority_receipt.json", receipt)
        if snapshot(files, trees, stable_absences) != stable_pre:
            raise RuntimeError("prepromote drift")
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        os.replace(prep, root)
        fsync_dir(root.parent)
        final_tree = rooted_tree(root)
        if (final_tree["file_count"] != 1
                or final_tree["inventory"] != [["authority_receipt.json",
                                                sha(root / "authority_receipt.json"),
                                                (root / "authority_receipt.json").stat().st_size]]
                or snapshot(files, trees, stable_absences) != stable_pre):
            raise RuntimeError("postpromote drift")
        committed = True
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        print(json.dumps({"path": str(root / "authority_receipt.json"),
                          "sha256": sha(root / "authority_receipt.json"),
                          "logical_bytes": (root / "authority_receipt.json").stat().st_size,
                          "authority_registration_tree": final_tree,
                          "committed_success": True, "deferred_signal": pending_signal["value"],
                          "outer_execution_wrapper_invocations": 0,
                          "corrected_inner_wrapper_invocations": 0,
                          "reconciler_r2_invocations": 0}, sort_keys=True))
        return 0
    except BaseException:
        if not committed and identity is not None:
            cleanup_owned(prep, identity)
        raise
    finally:
        if old_mask is not None and hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if not committed:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
