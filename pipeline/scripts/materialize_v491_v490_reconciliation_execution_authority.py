#!/usr/bin/env python3
"""Atomically register a v491 outer-wrapper authority; never run either wrapper or r2."""
from __future__ import annotations

import argparse
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
CONTRACT_PATH = ROOT / "pipeline/scripts/v491_v490_reconciliation_execution_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v491-v490-reconciliation-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v491-v490-reconciliation-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_v490_reconciliation_wrapper_attempt"
AUTHORITY_ROOT = J / "v491_v490_reconciliation_execution_authority_seed1633_20260825"
OUTER_EVIDENCE_ROOT = J / "v491_v490_reconciliation_outer_execution_evidence_seed1633_20260825"
INNER_ATTEMPT_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SOURCE_ROLES = {
    "rc2_forensic_source", "corrected_transport_script", "v490_authority_contract",
    "v490_authority_materializer", "v490_wrapper", "reconciler_r2",
    "outer_execution_wrapper",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_closure_sha256",
    "authority_materializer_source", "rc2_forensic_contract", "v490_authority_receipt",
    "v490_authority_registration_tree", "v490_helper_process_receipt",
    "v490_helper_evidence_tree", "repair_formal_registration_tree",
    "postregistration_static_registration_tree", "f813_registration_tree",
    "historical_absences", "current_absences_after_authority",
    "authority_receipt_contract", "execution_boundary",
}
FORENSIC_TOP_KEYS = {
    "format", "status", "occurred", "passed", "native_persistent_capture_available",
    "evidence_class", "failure_stage", "native_exit_code", "transport",
    "tool_capture_disclosure", "reconstructed_observations", "state_at_event",
    "source_records", "current_crosscheck", "unsafe_authorization",
}
AUTH_TOP_KEYS = {
    "format", "status", "passed", "authority_design_contract",
    "authority_materializer_source", "outer_execution_wrapper_source",
    "rc2_forensic_source", "rc2_forensic_copy", "corrected_transport_script_source",
    "corrected_transport_script_copy", "v490_authority_contract",
    "v490_authority_materializer_source", "v490_wrapper_source", "reconciler_r2_source",
    "v490_authority_receipt", "v490_authority_registration_tree",
    "v490_helper_process_receipt", "v490_helper_evidence_tree",
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
    "outer_wrapper_current", "rc2_forensic_source_current", "rc2_forensic_schema_exact",
    "rc2_zero_state_nonconsuming", "corrected_script_current", "forensic_copies_exact",
    "v490_authority_contract_current", "v490_authority_materializer_current",
    "v490_wrapper_current", "r2_current", "v490_authority_receipt_exact",
    "v490_authority_tree_exact1", "v490_authority_checks28_all_true",
    "v490_helper_process_exact", "v490_helper_evidence_exact6",
    "v490_helper_success_partition", "repair_formal_exact1", "static_exact1",
    "f813_exact2", "historical_absences", "current_absences", "no_live_process",
    "gpu_empty", "input_snapshots_equal", "execution_boundary",
})
AUTHORIZATION = {
    "outer_execution_wrapper_authorized": True,
    "outer_attempts_authorized": 1,
    "outer_attempts_consumed": 0,
    "retry_authorized": False,
    "direct_v490_wrapper_authorized": False,
    "direct_r2_authorized": False,
    "nested_v490_wrapper_invocations_authorized": 1,
    "nested_r2_invocations_authorized": 1,
    "nested_v490_wrapper_only_via_outer": True,
    "nested_r2_only_via_v490_wrapper": True,
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
    "v490_wrapper_executed": False,
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
    "v490_wrapper_invocations": 0,
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


def validate_forensic(value: dict, contract: dict, sources: dict[str, dict]) -> None:
    schema = contract["rc2_forensic_contract"]
    if (set(schema) != {"format", "status", "top_keys"}
            or set(value) != FORENSIC_TOP_KEYS or set(value) != set(schema["top_keys"])
            or value.get("format") != schema["format"] or value.get("status") != schema["status"]
            or value.get("occurred") is not True or value.get("passed") is not False
            or value.get("native_persistent_capture_available") is not False
            or value.get("native_exit_code") != 2):
        raise RuntimeError("rc2 forensic schema")
    observed = value.get("reconstructed_observations", {})
    if (observed.get("write_command_count") != 0
            or observed.get("fresh_helper_entry_count") != 0
            or observed.get("authority_materializer_invocation_count") != 0
            or observed.get("v490_wrapper_invocation_count") != 0
            or observed.get("r2_invocation_count") != 0
            or observed.get("attempt_consumed") is not False
            or observed.get("retry_consumed") is not False):
        raise RuntimeError("rc2 nonconsuming boundary")
    disclosure = value.get("tool_capture_disclosure", {})
    if (disclosure.get("native_argv_bytes") is not None
            or disclosure.get("native_stdout_bytes") is not None
            or disclosure.get("native_stderr_bytes") is not None
            or disclosure.get("native_stream_partition_available") is not False
            or disclosure.get("native_wall_seconds") is not None
            or disclosure.get("wrong_split_target") != {
                "path": "/root/autodl-tmp/IROS_WAM_2.0",
                "sha256": "46f4088cdf49cc144ec0897620c1e9639bd320e113a81b1e25919ef4e9e607aa",
                "logical_bytes": 144,
            }
            or disclosure.get("binary_operator_expected_observation_count") != 4
            or disclosure.get("four_test_exit_codes") != [2, 2, 2, 2]):
        raise RuntimeError("rc2 disclosure")
    if any(item is not False for item in value.get("unsafe_authorization", {}).values()):
        raise RuntimeError("rc2 unsafe authorization")
    state = value.get("state_at_event", {})
    state_true = {
        "fresh_helper_target_absent", "authority_root_absent", "authority_prep_absent",
        "authority_materialization_evidence_root_absent", "inner_attempt_root_absent",
        "inner_attempt_prep_absent", "transparent_output_absent", "qualification_root_absent",
    }
    if (any(state.get(key) is not True for key in state_true)
            or state.get("live_relevant_processes") != [] or state.get("gpu_compute_processes") != []):
        raise RuntimeError("rc2 event zero state")
    for source_record in value.get("source_records", {}).values():
        record(source_record)
    aliases = {
        "corrected_script": "corrected_transport_script",
        "v490_authority_contract": "v490_authority_contract",
        "v490_authority_materializer": "v490_authority_materializer",
        "v490_wrapper": "v490_wrapper",
        "r2": "reconciler_r2",
    }
    for forensic_name, source_name in aliases.items():
        if value["source_records"].get(forensic_name) != sources[source_name]:
            raise RuntimeError(f"rc2 source alias: {forensic_name}")


def synthetic_self_test() -> int:
    checks = {
        "authority_top_count": len(AUTH_TOP_KEYS) == 38,
        "check_count": len(AUTH_CHECK_KEYS) == 27,
        "source_roles": len(SOURCE_ROLES) == 7,
        "authorization_boundary": (
            AUTHORIZATION["outer_execution_wrapper_authorized"] is True
            and AUTHORIZATION["direct_v490_wrapper_authorized"] is False
            and AUTHORIZATION["direct_r2_authorized"] is False
            and AUTHORIZATION["retry_authorized"] is False
        ),
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
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1633
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

    forensic = json.loads(Path(sources["rc2_forensic_source"]["path"]).read_text())
    validate_forensic(forensic, contract, sources)
    v490_contract = json.loads(Path(sources["v490_authority_contract"]["path"]).read_text())
    v490_receipt_record = record(contract["v490_authority_receipt"])
    v490_receipt = json.loads(Path(v490_receipt_record["path"]).read_text())
    v490_schema = v490_contract["authority_receipt_contract"]
    if (set(v490_receipt) != set(v490_schema["top_keys"])
            or v490_receipt.get("format") != v490_schema["format"]
            or v490_receipt.get("status") != v490_schema["status"]
            or v490_receipt.get("passed") is not True
            or v490_receipt.get("checks") != {key: True for key in v490_schema["check_keys"]}
            or v490_receipt.get("checks_sha256") != v490_schema["checks_sha256"]
            or v490_receipt.get("authorization") != v490_schema["authorization_exact"]
            or v490_receipt.get("runtime_observation") != v490_schema["runtime_observation_exact"]):
        raise RuntimeError("v490 authority receipt")
    v490_authority_tree = verify_tree(contract["v490_authority_registration_tree"])
    helper_tree = verify_tree(contract["v490_helper_evidence_tree"])
    helper_record = record(contract["v490_helper_process_receipt"])
    helper = json.loads(Path(helper_record["path"]).read_text())
    if (helper.get("passed") is not True or helper.get("status") != "passed_exact_once_no_wrapper_or_r2_execution"
            or helper.get("materializer_invocations") != 1 or helper.get("wrapper_invocations") != 0
            or helper.get("r2_invocations") != 0 or helper.get("retry_authorized") is not False
            or helper.get("cleanup", {}).get("group_empty") is not True
            or helper.get("pre_post_snapshots_exactly_equal") is not True):
        raise RuntimeError("v490 helper process")
    repair_tree = verify_tree(contract["repair_formal_registration_tree"])
    static_tree = verify_tree(contract["postregistration_static_registration_tree"])
    f813_tree = verify_tree(contract["f813_registration_tree"])
    if (v490_authority_tree["file_count"] != 1 or helper_tree["file_count"] != 6
            or repair_tree["file_count"] != 1 or static_tree["file_count"] != 1
            or f813_tree["file_count"] != 2):
        raise RuntimeError("ancestry tree cardinality")
    if forensic["current_crosscheck"].get("v490_authority_tree") != v490_authority_tree:
        raise RuntimeError("forensic authority tree alias")
    if forensic["current_crosscheck"].get("authority_helper_evidence_tree") != helper_tree:
        raise RuntimeError("forensic helper tree alias")
    if (forensic["current_crosscheck"].get("v490_authority_receipt") != v490_receipt_record
            or forensic["current_crosscheck"].get("authority_helper_process_receipt") != helper_record
            or forensic["current_crosscheck"].get("inner_attempt_root_absent") is not True
            or forensic["current_crosscheck"].get("inner_attempt_prep_absent") is not True
            or forensic["current_crosscheck"].get("transparent_output_absent") is not True
            or forensic["current_crosscheck"].get("relevant_processes") != []
            or forensic["current_crosscheck"].get("gpu_compute_processes") != []):
        raise RuntimeError("forensic current crosscheck")
    if (v490_authority_tree["inventory"]
            != [["authority_receipt.json", v490_receipt_record["sha256"],
                 v490_receipt_record["logical_bytes"]]]
            or not any(row == ["process_receipt.json", helper_record["sha256"],
                               helper_record["logical_bytes"]] for row in helper_tree["inventory"])):
        raise RuntimeError("authority/helper tree record aliases")

    lineage = contract["lineage"]
    expected_lineage = {
        "execution_authority_root": str(AUTHORITY_ROOT),
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
    if (root != AUTHORITY_ROOT or root != root.resolve() or root.parent.is_symlink()
            or not root.parent.is_dir()):
        raise RuntimeError("authority root")
    historical_keys = {
        "superseded_static_root", "superseded_static_prep", "v489_superseded_static_root",
        "v489_superseded_static_prep", "fresh_static_prep", "v490_authority_prep",
        "inner_attempt_root", "inner_attempt_prep", "transparent_output", "transparent_tmp",
        "qualification_root", "execution_authority_root", "execution_authority_prep",
        "outer_evidence_root", "outer_evidence_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = decode_absences(contract["historical_absences"], historical_keys)
    current = decode_absences(contract["current_absences_after_authority"], current_keys)
    if (historical["execution_authority_root"] != root
            or historical["execution_authority_prep"] != prep
            or {key: historical[key] for key in current_keys} != current
            or any(os.path.lexists(path) for path in historical.values())):
        raise RuntimeError("absence prestate")
    live_fragments = {sources["outer_execution_wrapper"]["path"], sources["v490_wrapper"]["path"],
                      sources["reconciler_r2"]["path"]}
    if live_processes(live_fragments) or gpu_processes():
        raise RuntimeError("process or GPU prestate")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source),
             "v490_authority_receipt": v490_receipt_record["path"],
             "v490_helper_process_receipt": helper_record["path"]}
    for name, value in sources.items():
        files[f"source::{name}"] = value["path"]
    trees = {
        "v490_authority_REG": contract["v490_authority_registration_tree"]["root"],
        "v490_helper_evidence": contract["v490_helper_evidence_tree"]["root"],
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
    if pre != post or historical_pre["files"] != pre["files"] or historical_pre["trees"] != pre["trees"]:
        raise RuntimeError("input drift")

    forensic_copy_rel = "immutable_evidence/prehelper_rc2_transport_forensic_reconstructed.json"
    script_copy_rel = "immutable_evidence/v490_authority_materialize_corrected.sh"
    forensic_copy = {"path": str(root / forensic_copy_rel),
                     "sha256": sources["rc2_forensic_source"]["sha256"],
                     "logical_bytes": sources["rc2_forensic_source"]["logical_bytes"]}
    script_copy = {"path": str(root / script_copy_rel),
                   "sha256": sources["corrected_transport_script"]["sha256"],
                   "logical_bytes": sources["corrected_transport_script"]["logical_bytes"]}
    schema = contract["authority_receipt_contract"]
    checks = {key: True for key in AUTH_CHECK_KEYS}
    receipt = {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": materializer_record,
        "outer_execution_wrapper_source": sources["outer_execution_wrapper"],
        "rc2_forensic_source": sources["rc2_forensic_source"], "rc2_forensic_copy": forensic_copy,
        "corrected_transport_script_source": sources["corrected_transport_script"],
        "corrected_transport_script_copy": script_copy,
        "v490_authority_contract": sources["v490_authority_contract"],
        "v490_authority_materializer_source": sources["v490_authority_materializer"],
        "v490_wrapper_source": sources["v490_wrapper"], "reconciler_r2_source": sources["reconciler_r2"],
        "v490_authority_receipt": v490_receipt_record,
        "v490_authority_registration_tree": v490_authority_tree,
        "v490_helper_process_receipt": helper_record, "v490_helper_evidence_tree": helper_tree,
        "repair_formal_registration_tree": repair_tree,
        "postregistration_static_registration_tree": static_tree, "f813_registration_tree": f813_tree,
        "source_closure": sources, "source_closure_sha256": csha(sources),
        "outer_evidence_root": str(OUTER_EVIDENCE_ROOT), "inner_attempt_root": str(INNER_ATTEMPT_ROOT),
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
        raise ControlledSignal(signum)
    for sig in old_handlers:
        signal.signal(sig, on_signal)
    try:
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        prep.mkdir()
        identity = directory_identity(prep)
        evidence_dir = prep / "immutable_evidence"
        evidence_dir.mkdir()
        fsync_dir(prep)
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        copy_fsync(Path(sources["rc2_forensic_source"]["path"]), evidence_dir / Path(forensic_copy_rel).name)
        copy_fsync(Path(sources["corrected_transport_script"]["path"]), evidence_dir / Path(script_copy_rel).name)
        fsync_dir(evidence_dir)
        atomic_json(prep / "authority_receipt.json", receipt)
        if (regular(evidence_dir / Path(forensic_copy_rel).name,
                    forensic_copy["sha256"], forensic_copy["logical_bytes"])["sha256"] != forensic_copy["sha256"]
                or regular(evidence_dir / Path(script_copy_rel).name,
                           script_copy["sha256"], script_copy["logical_bytes"])["sha256"] != script_copy["sha256"]
                or snapshot(files, trees, stable_absences) != stable_pre):
            raise RuntimeError("prepromote drift")
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        os.replace(prep, root)
        committed = True
        fsync_dir(root.parent)
        final_tree = rooted_tree(root)
        if (final_tree["file_count"] != 3
                or [row[0] for row in final_tree["inventory"]]
                != ["authority_receipt.json", forensic_copy_rel, script_copy_rel]
                or regular(root / forensic_copy_rel, forensic_copy["sha256"], forensic_copy["logical_bytes"])
                   != forensic_copy
                or regular(root / script_copy_rel, script_copy["sha256"], script_copy["logical_bytes"])
                   != script_copy
                or snapshot(files, trees, current) != post):
            raise RuntimeError("postpromote drift")
        if old_mask is not None:
            if hasattr(signal, "sigpending"):
                pending = signal.sigpending()
                for candidate in (signal.SIGINT, signal.SIGTERM):
                    if candidate in pending:
                        pending_signal["value"] = candidate
                        break
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        print(json.dumps({"path": str(root / "authority_receipt.json"),
                          "sha256": sha(root / "authority_receipt.json"),
                          "logical_bytes": (root / "authority_receipt.json").stat().st_size,
                          "authority_registration_tree": final_tree,
                          "committed_success": True,
                          "deferred_signal": pending_signal["value"],
                          "outer_execution_wrapper_invocations": 0,
                          "v490_wrapper_invocations": 0,
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
