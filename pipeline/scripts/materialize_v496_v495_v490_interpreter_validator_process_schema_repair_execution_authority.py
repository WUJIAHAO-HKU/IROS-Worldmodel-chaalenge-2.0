#!/usr/bin/env python3
"""Atomically register the v496 process-schema repair authority; run no execution source."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CONTRACT_PATH = ROOT / "pipeline/scripts/v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v496-v495-v490-interpreter-validator-process-schema-repair-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v496-v495-v490-interpreter-validator-process-schema-repair-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_v496_interpreter_validator_process_schema_repaired_outer_attempt"
AUTHORITY_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_seed1638_20260825"
OUTER_EVIDENCE_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer_execution_evidence_seed1638_20260825"
INNER_ATTEMPT_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner_attempt_seed1638_20260825"
FAILED_V493_OUTER_ROOT = J / "v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825"
V493_AUTHORITY_ROOT = J / "v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825"
V493_FAILED_INNER_ROOT = J / "v493_v492_v490_corrected_reconciliation_inner_attempt_seed1635_20260825"
V493_PROCESS_PATH = J / "v493_v492_v490_corrected_reconciliation_execution_authority_materialization_evidence_seed1635_20260825/process_receipt.json"
FAILED_V494_AUTHORITY_ROOT = J / "v494_v493_v490_interpreter_symlink_repair_execution_authority_seed1636_20260825"
FAILED_V494_OUTER_ROOT = J / "v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer_execution_evidence_seed1636_20260825"
FAILED_V494_INNER_ROOT = J / "v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner_attempt_seed1636_20260825"
FAILED_V495_AUTHORITY_ROOT = J / "v495_v494_v490_interpreter_validator_repair_execution_authority_seed1637_20260825"
FAILED_V495_OUTER_ROOT = J / "v495_v494_v490_interpreter_validator_repaired_reconciliation_outer_execution_evidence_seed1637_20260825"
FAILED_V495_INNER_ROOT = J / "v495_v494_v490_interpreter_validator_repaired_reconciliation_inner_attempt_seed1637_20260825"
FAILED_V495_MATERIALIZATION_ROOT = J / "v495_v494_v490_interpreter_validator_repair_execution_authority_materialization_evidence_seed1637_20260825"
OLD_V490_INNER_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SOURCE_ROLES = {
    "v493_authority_contract", "v493_authority_materializer", "v493_outer_wrapper",
    "v493_inner_wrapper", "v494_authority_contract", "v494_authority_materializer",
    "v494_outer_wrapper", "v494_inner_wrapper", "corrected_inner_wrapper",
    "v495_authority_contract", "v495_authority_materializer",
    "v495_outer_wrapper", "v495_inner_wrapper",
    "reconciler_r2", "outer_execution_wrapper", "phase_a_design_contract",
    "v494_failure_forensic", "v494_failure_transport_script",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_closure_sha256",
    "authority_materializer_source", "v493_authority_contract", "v493_authority_receipt",
    "v493_authority_registration_tree", "v493_authority_materialization_evidence_tree",
    "v493_authority_materializer_process_receipt", "v493_outer_failure_tree",
    "v493_outer_terminal_receipt", "v493_outer_failure_ancestry",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "phase_a_design_contract_record", "historical_absences",
    "current_absences_after_authority", "authority_receipt_contract", "execution_boundary",
    "execution_interpreter_contract",
    "v494_authority_contract", "v494_materializer_failure_forensic",
    "v494_materializer_failure_transport_script", "v494_materializer_failure_ancestry",
    "v495_authority_contract", "v495_materializer_failure_tree",
    "v495_materializer_failure_process_receipt", "v495_materializer_failure_ancestry",
    "v493_authority_materializer_process_schema",
}
AUTH_TOP_KEYS = {
    "format", "status", "passed", "authority_design_contract",
    "authority_materializer_source", "outer_execution_wrapper_source",
    "corrected_inner_wrapper_source", "v493_inner_wrapper_source",
    "phase_a_design_contract_source", "reconciler_r2_source",
    "v493_authority_contract", "v493_authority_materializer_source",
    "v493_outer_wrapper_source", "v493_authority_receipt",
    "v493_authority_registration_tree", "v493_authority_materialization_evidence_tree",
    "v493_authority_materializer_process_receipt", "failed_v493_outer_execution_tree",
    "failed_v493_outer_terminal_receipt",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "source_closure", "source_closure_sha256",
    "outer_evidence_root", "inner_attempt_root", "transparent_static_receipt_path",
    "historical_absences", "required_absences", "checks", "check_keys",
    "check_key_set_sha256", "checks_sha256", "input_pre_snapshot",
    "input_post_snapshot", "input_snapshots_exactly_equal", "authorization",
    "runtime_observation", "execution_boundary", "execution_interpreter_evidence",
    "v494_authority_contract_source", "v494_authority_materializer_source",
    "v494_outer_wrapper_source", "v494_inner_wrapper_source",
    "v494_materializer_failure_forensic", "v494_materializer_failure_transport_script",
    "v494_materializer_failure_ancestry",
    "v495_authority_contract_source", "v495_authority_materializer_source",
    "v495_outer_wrapper_source", "v495_inner_wrapper_source",
    "v495_materializer_failure_tree", "v495_materializer_failure_process_receipt",
    "v495_materializer_failure_ancestry", "v493_authority_materializer_process_schema",
}
AUTH_CHECK_KEYS = sorted({
    "authority_contract_current", "authority_materializer_current",
    "corrected_inner_current", "corrected_inner_diff_exact", "current_absences",
    "execution_boundary", "f813_exact2", "failed_v493_outer_exact4",
    "failed_v493_outer_no_retry_partition", "failed_v493_outer_terminal_exact",
    "gpu_empty", "historical_absences", "input_snapshots_equal", "no_live_process",
    "outer_wrapper_current", "phase_a_contract_current", "phase_a_contract_spec_exact2",
    "postregistration_static_exact1", "r2_current", "repair_formal_exact1",
    "source_closure_current", "v493_authority_contract_current", "v493_authority_exact3",
    "v493_authority_materialization_evidence_exact6",
    "v493_authority_materializer_process_exact", "v493_outer_wrapper_current",
    "execution_interpreter_chain_exact", "execution_interpreter_runtime_exact",
    "v494_authority_contract_current", "v494_authority_materializer_current",
    "v494_inner_wrapper_current", "v494_outer_wrapper_current",
    "v494_failure_forensic_exact", "v494_failure_transport_script_current",
    "v494_materializer_failure_partition_exact", "validator_torch_version_scope_exact",
    "validator_torch_version_tamper_suite_passed",
    "v495_authority_contract_current", "v495_authority_materializer_current",
    "v495_inner_wrapper_current", "v495_outer_wrapper_current",
    "v495_materializer_failure_exact6", "v495_materializer_failure_process_exact",
    "v493_process_schema_exact24", "validator_process_schema_tamper_suite_passed",
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
V493_PROCESS_KEYS = {
    "argv", "authority_receipt", "authority_tree", "cleanup",
    "corrected_inner_wrapper_invocations", "failed_v492_outer_terminal_receipt",
    "format", "helper_returncode", "intent", "materializer_invocations",
    "materializer_returncode", "outer_wrapper_invocations", "passed",
    "post_snapshot", "pre_post_snapshots_exactly_equal", "pre_snapshot",
    "r2_invocations", "retry_authorized", "status", "stderr", "stdout",
    "transport_helper", "transport_helper_copy", "wall_seconds",
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


def execution_interpreter_evidence() -> dict:
    lexical_path = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
    intermediate_path = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python")
    resolved_path = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")

    def lstat_row(path: Path) -> dict:
        observed = os.lstat(path)
        return {"device": observed.st_dev, "inode": observed.st_ino,
                "mode": observed.st_mode, "size": observed.st_size}

    if sys.executable != str(lexical_path):
        raise RuntimeError("interpreter lexical executable")
    lexical_lstat = lstat_row(lexical_path)
    lexical_target = os.readlink(lexical_path)
    intermediate_lstat = lstat_row(intermediate_path)
    intermediate_target = os.readlink(intermediate_path)
    resolved_lstat = lstat_row(resolved_path)
    if (not stat.S_ISLNK(lexical_lstat["mode"])
            or lexical_target != str(intermediate_path)
            or not stat.S_ISLNK(intermediate_lstat["mode"])
            or intermediate_target != "python3.11"
            or (intermediate_path.parent / intermediate_target).resolve(strict=True) != resolved_path
            or stat.S_ISLNK(resolved_lstat["mode"])
            or not stat.S_ISREG(resolved_lstat["mode"])):
        raise RuntimeError("interpreter chain drift")
    import numpy
    import torch
    evidence = {
        "lexical": {"path": str(lexical_path), "lstat": lexical_lstat,
                    "readlink": lexical_target},
        "intermediate": {"path": str(intermediate_path), "lstat": intermediate_lstat,
                         "readlink": intermediate_target},
        "resolved": {**regular(resolved_path,
                               "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788",
                               25555040), "lstat": resolved_lstat},
        "runtime": {"sys_executable": sys.executable, "python_version": sys.version,
                    "numpy_version": numpy.__version__, "torch_version": torch.__version__},
    }
    expected = {
        "lexical": {"path": str(lexical_path),
                    "lstat": {"device": 2304, "inode": 17217118070,
                              "mode": 41471, "size": 49},
                    "readlink": str(intermediate_path)},
        "intermediate": {"path": str(intermediate_path),
                         "lstat": {"device": 2304, "inode": 7529246267,
                                   "mode": 41471, "size": 10},
                         "readlink": "python3.11"},
        "resolved": {"path": str(resolved_path),
                     "sha256": "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788",
                     "logical_bytes": 25555040,
                     "lstat": {"device": 2304, "inode": 7529484239,
                               "mode": 33277, "size": 25555040}},
        "runtime": {"sys_executable": str(lexical_path),
                    "python_version": "3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]",
                    "numpy_version": "1.26.4", "torch_version": "2.7.0+cu128"},
    }
    if evidence != expected:
        raise RuntimeError("interpreter evidence drift")
    return evidence


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
    value = {"files": file_rows, "trees": tree_rows, "absences": absence_rows,
             "execution_interpreter_evidence": execution_interpreter_evidence()}
    value["snapshot_sha256"] = csha(value)
    return value


def validate_torch_version_probe_ast(parsed: ast.Module) -> None:
    functions = [node for node in ast.walk(parsed)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "execution_interpreter_evidence"]
    if len(functions) != 1:
        raise RuntimeError("torch version probe function")
    function = functions[0]
    parents = {child: parent for parent in ast.walk(parsed) for child in ast.iter_child_nodes(parent)}
    torch_imports = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_imports.append((node, alias))
        elif isinstance(node, ast.ImportFrom) and (
                (node.module and node.module.split(".")[0] == "torch")
                or any(alias.name.split(".")[0] == "torch" for alias in node.names)):
            raise RuntimeError("torch import-from forbidden")
    if (len(torch_imports) != 1
            or len(torch_imports[0][0].names) != 1
            or torch_imports[0][1].name != "torch"
            or torch_imports[0][1].asname is not None
            or parents.get(torch_imports[0][0]) is not function):
        raise RuntimeError("torch import scope")
    torch_names = [node for node in ast.walk(parsed)
                   if isinstance(node, ast.Name) and node.id == "torch"]
    torch_attributes = [node for node in ast.walk(parsed)
                        if isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name) and node.value.id == "torch"]
    if (len(torch_names) != 1 or len(torch_attributes) != 1
            or torch_attributes[0].attr != "__version__"
            or not isinstance(torch_attributes[0].ctx, ast.Load)
            or not isinstance(torch_names[0].ctx, ast.Load)
            or torch_attributes[0].value is not torch_names[0]
            or parents.get(torch_attributes[0]) is None
            or function not in list(ast.iter_child_nodes(parsed))):
        raise RuntimeError("torch version-only load")
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) and any(
                isinstance(child, ast.Name) and child.id == "torch" for child in ast.walk(node)):
            raise RuntimeError("torch call forbidden")


def validate_v493_process_receipt(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != V493_PROCESS_KEYS:
        raise RuntimeError("v493 process exact24 keyset")
    integer_partition = {
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "r2_invocations": 0,
    }
    if any(type(value.get(key)) is not int or value.get(key) != expected
           for key, expected in integer_partition.items()):
        raise RuntimeError("v493 process integer partition")
    if (value.get("status") != "passed_exact_once_no_outer_or_inner_execution"
            or value.get("passed") is not True
            or value.get("retry_authorized") is not False
            or value.get("pre_post_snapshots_exactly_equal") is not True
            or value.get("cleanup", {}).get("reaped") is not True
            or value.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v493 process semantics")


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
            or "defexecution_interpreter_evidence()" not in normalized
            or "EXECUTION_INTERPRETER_CONTRACT=" not in normalized
            or "regular(Path(sys.executable))" in normalized
            or "regular(RLPY)" in normalized
            or "os.readlink(RLPY)" not in normalized
            or "os.readlink(intermediate)" not in normalized
            or "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64" not in text
            or "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788" not in text
            or "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377" not in text):
        raise RuntimeError("corrected inner interpreter and phase-contract repair")
    validate_torch_version_probe_ast(parsed)
    imported = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    if imported & {"transformers", "rlinf"}:
        raise RuntimeError("corrected inner forbidden import")


def synthetic_self_test() -> int:
    checks = {
        "authority_top_count": len(AUTH_TOP_KEYS) == 55,
        "check_count": len(AUTH_CHECK_KEYS) == 45,
        "source_roles": len(SOURCE_ROLES) == 18,
        "authorization_boundary": (
            AUTHORIZATION["outer_execution_wrapper_authorized"] is True
            and AUTHORIZATION["direct_corrected_inner_authorized"] is False
            and AUTHORIZATION["direct_r2_authorized"] is False
            and AUTHORIZATION["retry_authorized"] is False
        ),
        "fresh_roots": all("v496_" in path.name for path in
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
    valid_probe = ast.parse("def execution_interpreter_evidence():\n import torch\n return torch.__version__\n")
    try:
        validate_torch_version_probe_ast(valid_probe)
        checks["validator_exact_version_probe"] = True
    except RuntimeError:
        checks["validator_exact_version_probe"] = False
    tampered = [
        "import torch\ndef execution_interpreter_evidence():\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n import torch\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch as t\n return t.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n from helper import torch\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n return torch.cuda\n",
        "def execution_interpreter_evidence():\n import torch\n return (torch.__version__,torch.__version__)\n",
        "def execution_interpreter_evidence():\n import torch\n return torch\n",
        "def execution_interpreter_evidence():\n import torch\n return torch.cuda()\n",
    ]
    rejected = []
    for source in tampered:
        try:
            validate_torch_version_probe_ast(ast.parse(source))
            rejected.append(False)
        except RuntimeError:
            rejected.append(True)
    checks["validator_tamper_suite"] = all(rejected)
    if V493_PROCESS_PATH.is_file():
        actual_process = json.loads(V493_PROCESS_PATH.read_text())
        try:
            validate_v493_process_receipt(actual_process)
            checks["actual_v493_process_receipt"] = (
                regular(V493_PROCESS_PATH,
                        "babcfda7fad625ea7d0b7a34592e119d6287fa4b34dcfbda743f7473a0fe4789",
                        19613)["logical_bytes"] == 19613)
        except RuntimeError:
            checks["actual_v493_process_receipt"] = False
        variants = []
        missing = dict(actual_process); missing.pop("corrected_inner_wrapper_invocations"); variants.append(missing)
        stale = dict(actual_process); stale["inner_wrapper_invocations"] = stale.pop("corrected_inner_wrapper_invocations"); variants.append(stale)
        both = dict(actual_process); both["inner_wrapper_invocations"] = 0; variants.append(both)
        extra = dict(actual_process); extra["unexpected"] = 0; variants.append(extra)
        nonzero = dict(actual_process); nonzero["corrected_inner_wrapper_invocations"] = 1; variants.append(nonzero)
        boolean = dict(actual_process); boolean["corrected_inner_wrapper_invocations"] = False; variants.append(boolean)
        process_rejected = []
        for variant in variants:
            try:
                validate_v493_process_receipt(variant)
                process_rejected.append(False)
            except RuntimeError:
                process_rejected.append(True)
        checks["process_schema_tamper_suite"] = all(process_rejected)
    else:
        checks["actual_v493_process_receipt"] = True
        checks["process_schema_tamper_suite"] = True
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
    parser.add_argument("--read-only-preflight", action="store_true")
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1638
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
    v494_expected = {
        "v494_authority_contract": ("2f4cef2671dfd5c443d5a7f53dc0ea7b770b520da84cba3512a363884d56bd4a", 29803),
        "v494_authority_materializer": ("5b9396fd74e1d161fa41610108d823f356313992364cba708ed869519fa0a421", 39236),
        "v494_inner_wrapper": ("1f2c9de60703f12c22966b7aeb3526a6c32e0d5a0e0258094c9d208b9a56f4bf", 79760),
        "v494_outer_wrapper": ("bea4b6df050afb22c833908725a6af7417b292984d8ddb4fbb972b99128d3df5", 43024),
    }
    if any((sources[role]["sha256"], sources[role]["logical_bytes"]) != expected
           for role, expected in v494_expected.items()):
        raise RuntimeError("v494 frozen sources")
    if record(contract["v494_authority_contract"]) != sources["v494_authority_contract"]:
        raise RuntimeError("v494 authority contract alias")
    v494_contract = json.loads(Path(sources["v494_authority_contract"]["path"]).read_text())
    if (len(v494_contract) != 24
            or v494_contract.get("format") != "strict-track2-v494-v493-v490-interpreter-symlink-repair-execution-authority-design-contract-v1"
            or v494_contract.get("status") != CONTRACT_STATUS
            or v494_contract.get("seed") != 1636
            or v494_contract.get("source_closure", {}).get("corrected_inner_wrapper") != sources["v494_inner_wrapper"]
            or v494_contract.get("source_closure", {}).get("outer_execution_wrapper") != sources["v494_outer_wrapper"]):
        raise RuntimeError("v494 authority contract schema")
    forensic_record = record(contract["v494_materializer_failure_forensic"])
    transport_record = record(contract["v494_materializer_failure_transport_script"])
    if (forensic_record != sources["v494_failure_forensic"]
            or transport_record != sources["v494_failure_transport_script"]
            or forensic_record["sha256"] != "248c4a94a24c3b793bb29b6cc07a5955a93694f7e22b1a9363ef5bb7ee0ba0dc"
            or forensic_record["logical_bytes"] != 6409
            or transport_record["sha256"] != "0d4669e9a1fa52ee98fe2c459e9e42d880116e4428270edcefb0f50e5b95d49c"
            or transport_record["logical_bytes"] != 942):
        raise RuntimeError("v494 failure sources")
    forensic = json.loads(Path(forensic_record["path"]).read_text())
    failure_ancestry = contract["v494_materializer_failure_ancestry"]
    expected_failure_ancestry = {
        "failed_status": "frozen_reconstructed_v494_materializer_prestate_failure_no_retry",
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "native_exit_code": 1,
        "native_persistent_capture_available": False,
        "tool_capture_sha256": "4a65e33de2de1da29e8a36d39e5374e43b66435aa5e6bc3722ccfca574fb4a4c",
        "tool_capture_logical_bytes": 787,
        "root_cause": "v494 validator blanket-forbade function-local torch import required for exact runtime version evidence",
        "corrected_rule": "allow exactly one no-alias torch import and one Load torch.__version__ inside execution_interpreter_evidence; reject every other torch import, name, attribute, and call",
        "retry_authorized": False,
    }
    old_source_projection = {
        "authority_contract": sources["v494_authority_contract"],
        "authority_materializer": sources["v494_authority_materializer"],
        "corrected_inner_wrapper": sources["v494_inner_wrapper"],
        "outer_execution_wrapper": sources["v494_outer_wrapper"],
    }
    invocation = forensic.get("invocation", {})
    tool_capture = forensic.get("tool_capture", {})
    state = forensic.get("state_after_failure", {})
    if (failure_ancestry != expected_failure_ancestry
            or forensic.get("format") != "strict-track2-v495-v494-authority-materializer-failure-forensic-reconstructed-v1"
            or forensic.get("status") != expected_failure_ancestry["failed_status"]
            or forensic.get("occurred") is not True or forensic.get("passed") is not False
            or forensic.get("native_exit_code") != 1
            or forensic.get("native_persistent_capture_available") is not False
            or forensic.get("source_records") != old_source_projection
            or invocation.get("materializer_invocations") != 1
            or any(invocation.get(key) != 0 for key in ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or invocation.get("transport_script", {}).get("sha256") != transport_record["sha256"]
            or invocation.get("transport_script", {}).get("logical_bytes") != transport_record["logical_bytes"]
            or tool_capture.get("combined_output_sha256") != expected_failure_ancestry["tool_capture_sha256"]
            or tool_capture.get("combined_output_logical_bytes") != expected_failure_ancestry["tool_capture_logical_bytes"]
            or "corrected inner forbidden import" not in tool_capture.get("combined_output_text", "")
            or state.get("all_absent") is not True
            or state.get("live_reconciliation_processes_empty") is not True
            or state.get("gpu_compute_apps_empty") is not True):
        raise RuntimeError("v494 materializer failure forensic")
    v495_expected = {
        "v495_authority_contract": ("60d978a803bd148f6fe68906e71113cf33e61a3ae5bf77e22c8f2855c68c2823", 37084),
        "v495_authority_materializer": ("b0f4b59212ea67754a12394bf1f5a8e7d2c5389c93c44207e0099d9f2850fe18", 50978),
        "v495_inner_wrapper": ("b52e7eb29870141bda85eead81cbc1c3a47522ad86e6a7b6829f7052766a8f76", 90261),
        "v495_outer_wrapper": ("3aae6deb4059e143cc40a3e10bdf6f1cf51e20e884c0459b4aa950e0d480d526", 50749),
    }
    if any((sources[role]["sha256"], sources[role]["logical_bytes"]) != expected
           for role, expected in v495_expected.items()):
        raise RuntimeError("v495 frozen sources")
    if record(contract["v495_authority_contract"]) != sources["v495_authority_contract"]:
        raise RuntimeError("v495 authority contract alias")
    v495_contract = json.loads(Path(sources["v495_authority_contract"]["path"]).read_text())
    if (len(v495_contract) != 28
            or v495_contract.get("format") != "strict-track2-v495-v494-v490-interpreter-validator-repair-execution-authority-design-contract-v1"
            or v495_contract.get("status") != CONTRACT_STATUS
            or v495_contract.get("seed") != 1637
            or v495_contract.get("authority_materializer_source") != sources["v495_authority_materializer"]
            or v495_contract.get("source_closure", {}).get("corrected_inner_wrapper") != sources["v495_inner_wrapper"]
            or v495_contract.get("source_closure", {}).get("outer_execution_wrapper") != sources["v495_outer_wrapper"]):
        raise RuntimeError("v495 authority contract schema")
    v495_failure_tree = verify_tree(contract["v495_materializer_failure_tree"])
    v495_failure_process_record = record(contract["v495_materializer_failure_process_receipt"])
    v495_failure_process = json.loads(Path(v495_failure_process_record["path"]).read_text())
    v495_failure_ancestry = contract["v495_materializer_failure_ancestry"]
    expected_v495_failure_ancestry = {
        "failed_status": "failed_no_retry",
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "native_exit_code": 1,
        "retry_authorized": False,
        "root_cause": "v495 materializer required stale inner_wrapper_invocations absent from immutable v493 process receipt",
        "corrected_rule": "validate exact24 process receipt with corrected_inner_wrapper_invocations int zero and reject stale, missing, both, extra, nonzero, and bool variants",
    }
    if (v495_failure_tree["root"] != str(FAILED_V495_MATERIALIZATION_ROOT)
            or v495_failure_tree["file_count"] != 6
            or v495_failure_tree["logical_file_bytes"] != 32139
            or v495_failure_tree["sha256sum_lines_digest_sha256"] != "21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1"
            or v495_failure_tree["canonical_json_triples_digest_sha256"] != "9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290"
            or not any(row == ["process_receipt.json", v495_failure_process_record["sha256"],
                               v495_failure_process_record["logical_bytes"]]
                       for row in v495_failure_tree["inventory"])
            or v495_failure_process_record["sha256"] != "72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4"
            or v495_failure_process_record["logical_bytes"] != 411):
        raise RuntimeError("v495 failure tree")
    if (v495_failure_ancestry != expected_v495_failure_ancestry
            or v495_failure_process.get("format") != "strict-track2-v495-authority-materializer-process-receipt-v1"
            or v495_failure_process.get("status") != "failed_no_retry"
            or v495_failure_process.get("passed") is not False
            or type(v495_failure_process.get("materializer_invocations")) is not int
            or v495_failure_process.get("materializer_invocations") != 1
            or any(type(v495_failure_process.get(key)) is not int or v495_failure_process.get(key) != 0
                   for key in ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "r2_invocations"))
            or v495_failure_process.get("retry_authorized") is not False
            or v495_failure_process.get("cleanup", {}).get("reaped") is not True
            or v495_failure_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v495 materializer failure ancestry")
    phase_record = sources["phase_a_design_contract"]
    if (phase_record != contract["phase_a_design_contract_record"]
            or phase_record["sha256"] != "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
            or phase_record["logical_bytes"] != 43960):
        raise RuntimeError("phase contract current record")
    validate_corrected_inner(Path(sources["corrected_inner_wrapper"]["path"]))
    interpreter = execution_interpreter_evidence()
    if contract["execution_interpreter_contract"] != interpreter:
        raise RuntimeError("execution interpreter contract")

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

    v493_contract = json.loads(Path(sources["v493_authority_contract"]["path"]).read_text())
    if record(contract["v493_authority_contract"]) != sources["v493_authority_contract"]:
        raise RuntimeError("v493 contract alias")
    v493_receipt_record = record(contract["v493_authority_receipt"])
    v493_receipt = json.loads(Path(v493_receipt_record["path"]).read_text())
    old_schema = v493_contract.get("authority_receipt_contract", {})
    if (set(v493_receipt) != set(old_schema.get("top_keys", []))
            or len(v493_receipt) != 39 or len(v493_receipt.get("checks", {})) != 26
            or v493_receipt.get("format") != old_schema.get("format")
            or v493_receipt.get("status") != old_schema.get("status")
            or v493_receipt.get("passed") is not True
            or v493_receipt.get("checks") != {key: True for key in old_schema.get("check_keys", [])}
            or v493_receipt.get("checks_sha256") != old_schema.get("checks_sha256")
            or v493_receipt.get("authorization") != old_schema.get("authorization_exact")
            or v493_receipt.get("runtime_observation") != old_schema.get("runtime_observation_exact")):
        raise RuntimeError("v493 authority receipt")
    v493_auth_tree = verify_tree(contract["v493_authority_registration_tree"])
    if (v493_auth_tree["file_count"] != 1
            or not any(row == ["authority_receipt.json", v493_receipt_record["sha256"],
                               v493_receipt_record["logical_bytes"]]
                       for row in v493_auth_tree["inventory"])):
        raise RuntimeError("v493 authority exact3")
    v493_evidence_tree = verify_tree(contract["v493_authority_materialization_evidence_tree"])
    v493_process_record = record(contract["v493_authority_materializer_process_receipt"])
    v493_process = json.loads(Path(v493_process_record["path"]).read_text())
    validate_v493_process_receipt(v493_process)
    process_schema = {
        "record": v493_process_record,
        "exact_keys": sorted(V493_PROCESS_KEYS),
        "key_set_sha256": csha(sorted(V493_PROCESS_KEYS)),
        "required_values": {
            "status": "passed_exact_once_no_outer_or_inner_execution",
            "passed": True,
            "materializer_invocations": 1,
            "outer_wrapper_invocations": 0,
            "corrected_inner_wrapper_invocations": 0,
            "r2_invocations": 0,
            "retry_authorized": False,
            "pre_post_snapshots_exactly_equal": True,
            "cleanup": {"reaped": True, "group_empty": True},
        },
        "forbidden_keys": ["inner_wrapper_invocations"],
    }
    if contract["v493_authority_materializer_process_schema"] != process_schema:
        raise RuntimeError("v493 process schema contract")
    if (v493_evidence_tree["file_count"] != 6
            or not any(row == ["process_receipt.json", v493_process_record["sha256"],
                               v493_process_record["logical_bytes"]]
                       for row in v493_evidence_tree["inventory"])
            or v493_process.get("passed") is not True):
        raise RuntimeError("v493 authority materialization evidence")

    failed_tree = verify_tree(contract["v493_outer_failure_tree"])
    failed_terminal_record = record(contract["v493_outer_terminal_receipt"])
    failed_terminal = json.loads(Path(failed_terminal_record["path"]).read_text())
    stderr_path = Path(failed_tree["root"]) / "corrected_inner_stderr.log"
    failure_spec = contract["v493_outer_failure_ancestry"]
    expected_failure_spec = {
        "failed_status": "failed_no_retry",
        "superseded_outer_wrapper_invocations": 1,
        "superseded_corrected_inner_wrapper_invocations": 1,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "root_cause": "superseded corrected inner required lexical RLPY symlink to be regular nonsymlink",
        "corrected_rule": "bind exact lexical and intermediate symlinks, resolved regular target, hash, bytes, and runtime versions",
    }
    if (failed_tree["file_count"] != 4 or failure_spec != expected_failure_spec
            or not any(row == ["terminal_receipt.json", failed_terminal_record["sha256"],
                               failed_terminal_record["logical_bytes"]]
                       for row in failed_tree["inventory"])
            or failed_terminal.get("status") != "failed_no_retry"
            or failed_terminal.get("passed") is not False
            or failed_terminal.get("nested_corrected_inner_invocations") != 1
            or failed_terminal.get("retry_authorized") is not False
            or failed_terminal.get("cleanup", {}).get("reaped") is not True
            or failed_terminal.get("cleanup", {}).get("group_empty") is not True
            or failed_terminal.get("transparent_present") is not False
            or "RuntimeError" not in stderr_path.read_text()
            or "regular: /root/autodl-tmp/conda_envs/rlinf_track2/bin/python" not in stderr_path.read_text()):
        raise RuntimeError("v493 outer failed-no-retry ancestry")

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
        "failed_v492_outer_evidence_prep", "v493_authority_prep",
        "failed_v493_outer_evidence_prep", "execution_authority_root",
        "failed_v493_inner_attempt_root", "failed_v493_inner_attempt_prep",
        "failed_v494_authority_root", "failed_v494_authority_prep",
        "failed_v494_outer_evidence_root", "failed_v494_outer_evidence_prep",
        "failed_v494_inner_attempt_root", "failed_v494_inner_attempt_prep",
        "failed_v495_authority_root", "failed_v495_authority_prep",
        "failed_v495_outer_evidence_root", "failed_v495_outer_evidence_prep",
        "failed_v495_inner_attempt_root", "failed_v495_inner_attempt_prep",
        "execution_authority_prep", "outer_evidence_root", "outer_evidence_prep",
        "corrected_inner_attempt_root", "corrected_inner_attempt_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = decode_absences(contract["historical_absences"], historical_keys)
    current = decode_absences(contract["current_absences_after_authority"], current_keys)
    if (historical["execution_authority_root"] != root
            or historical["execution_authority_prep"] != prep
            or historical["failed_v493_outer_evidence_prep"] != FAILED_V493_OUTER_ROOT.with_name(FAILED_V493_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v493_inner_attempt_root"] != V493_FAILED_INNER_ROOT
            or historical["failed_v493_inner_attempt_prep"] != V493_FAILED_INNER_ROOT.with_name(V493_FAILED_INNER_ROOT.name + ".attempt-prep")
            or historical["failed_v494_authority_root"] != FAILED_V494_AUTHORITY_ROOT
            or historical["failed_v494_authority_prep"] != FAILED_V494_AUTHORITY_ROOT.with_name(FAILED_V494_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["failed_v494_outer_evidence_root"] != FAILED_V494_OUTER_ROOT
            or historical["failed_v494_outer_evidence_prep"] != FAILED_V494_OUTER_ROOT.with_name(FAILED_V494_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v494_inner_attempt_root"] != FAILED_V494_INNER_ROOT
            or historical["failed_v494_inner_attempt_prep"] != FAILED_V494_INNER_ROOT.with_name(FAILED_V494_INNER_ROOT.name + ".attempt-prep")
            or historical["failed_v495_authority_root"] != FAILED_V495_AUTHORITY_ROOT
            or historical["failed_v495_authority_prep"] != FAILED_V495_AUTHORITY_ROOT.with_name(FAILED_V495_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["failed_v495_outer_evidence_root"] != FAILED_V495_OUTER_ROOT
            or historical["failed_v495_outer_evidence_prep"] != FAILED_V495_OUTER_ROOT.with_name(FAILED_V495_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v495_inner_attempt_root"] != FAILED_V495_INNER_ROOT
            or historical["failed_v495_inner_attempt_prep"] != FAILED_V495_INNER_ROOT.with_name(FAILED_V495_INNER_ROOT.name + ".attempt-prep")
            or historical["old_v490_inner_attempt_root"] != OLD_V490_INNER_ROOT
            or {key: historical[key] for key in current_keys} != current
            or any(os.path.lexists(path) for path in historical.values())):
        raise RuntimeError("absence prestate")
    live_fragments = {row["path"] for role, row in sources.items()
                      if role in {"corrected_inner_wrapper", "outer_execution_wrapper", "reconciler_r2"}}
    if live_processes(live_fragments) or gpu_processes():
        raise RuntimeError("process or GPU prestate")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source),
             "v493_authority_receipt": v493_receipt_record["path"],
             "v493_process_receipt": v493_process_record["path"],
             "failed_v493_terminal": failed_terminal_record["path"],
             "v495_failure_process": v495_failure_process_record["path"]}
    files.update({f"source::{name}": row["path"] for name, row in sources.items()})
    trees = {
        "v493_authority_REG": contract["v493_authority_registration_tree"]["root"],
        "v493_authority_materialization": contract["v493_authority_materialization_evidence_tree"]["root"],
        "failed_v493_outer": contract["v493_outer_failure_tree"]["root"],
        "v495_materializer_failure": contract["v495_materializer_failure_tree"]["root"],
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
        "v493_inner_wrapper_source": sources["v493_inner_wrapper"],
        "phase_a_design_contract_source": phase_record,
        "reconciler_r2_source": sources["reconciler_r2"],
        "v493_authority_contract": sources["v493_authority_contract"],
        "v493_authority_materializer_source": sources["v493_authority_materializer"],
        "v493_outer_wrapper_source": sources["v493_outer_wrapper"],
        "v494_authority_contract_source": sources["v494_authority_contract"],
        "v494_authority_materializer_source": sources["v494_authority_materializer"],
        "v494_outer_wrapper_source": sources["v494_outer_wrapper"],
        "v494_inner_wrapper_source": sources["v494_inner_wrapper"],
        "v494_materializer_failure_forensic": forensic_record,
        "v494_materializer_failure_transport_script": transport_record,
        "v494_materializer_failure_ancestry": failure_ancestry,
        "v495_authority_contract_source": sources["v495_authority_contract"],
        "v495_authority_materializer_source": sources["v495_authority_materializer"],
        "v495_outer_wrapper_source": sources["v495_outer_wrapper"],
        "v495_inner_wrapper_source": sources["v495_inner_wrapper"],
        "v495_materializer_failure_tree": v495_failure_tree,
        "v495_materializer_failure_process_receipt": v495_failure_process_record,
        "v495_materializer_failure_ancestry": v495_failure_ancestry,
        "v493_authority_materializer_process_schema": process_schema,
        "v493_authority_receipt": v493_receipt_record,
        "v493_authority_registration_tree": v493_auth_tree,
        "v493_authority_materialization_evidence_tree": v493_evidence_tree,
        "v493_authority_materializer_process_receipt": v493_process_record,
        "failed_v493_outer_execution_tree": failed_tree,
        "failed_v493_outer_terminal_receipt": failed_terminal_record,
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
        "execution_interpreter_evidence": interpreter,
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
    if args.read_only_preflight:
        print(json.dumps({"status": "passed_read_only_preflight_no_materialization",
                          "contract": contract_record,
                          "materializer": materializer_record,
                          "authority_root_absent": not os.path.lexists(root),
                          "authority_prep_absent": not os.path.lexists(prep),
                          "receipt_top_count": len(receipt),
                          "check_count": len(checks),
                          "source_count": len(sources),
                          "input_snapshots_exactly_equal": pre == post,
                          "execution_interpreter_evidence": interpreter}, sort_keys=True))
        return 0

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
