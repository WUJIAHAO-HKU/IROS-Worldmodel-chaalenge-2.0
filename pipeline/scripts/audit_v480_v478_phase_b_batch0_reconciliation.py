#!/usr/bin/env python3
"""Immutable reconciliation of the v478 Phase-B batch-0 audit JSON boundary."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PRE_FORMAT = "strict-track2-v480-v478-phase-b-batch0-reconciliation-preregistration-v1"
OUT_FORMAT = "strict-track2-v480-v478-phase-b-batch0-immutable-reconciliation-v1"
BATCH_AUDIT_FORMAT = "strict-track2-v478-public-train-temporal200-batch-audit-v1"
TREE_DIGEST = "0ff88bdbd6196d33d94796d7733fdc81bd2102a55a8c53d611c1a9454414d1b5"
TREE_LINES_DIGEST = "b9a44ba8b63c2cc721e6bd94d03d38efff61f18fa32fe437904faf88e3e2d031"
TREE_FILES = 41
TREE_BYTES = 201004510
EXPECTED_CHECKS = {
    "input_closure",
    "exact20_selection_order",
    "batch_exact_tree",
    "all_rows_independently_valid",
    "batch_report_integrity",
    "storage_formula_and_historic_gates",
    "diagnostic_recomputed_but_not_gate",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic(path: Path, value) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, default=canonical_numpy)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require(record: dict) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha(path) != record["sha256"]:
        raise RuntimeError(f"immutable evidence drift: {path}")
    return path


def inventory(root: Path) -> tuple[list[list[object]], str, str]:
    root = root.resolve()
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("batch tree symlink forbidden")
    items = [
        [path.relative_to(root).as_posix(), path.stat().st_size, sha(path)]
        for path in sorted(paths)
        if path.is_file()
    ]
    digest = hashlib.sha256(
        json.dumps([[x[0], x[2]] for x in items], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lines_digest = hashlib.sha256("".join(f"{x[2]}  {x[0]}\n" for x in items).encode()).hexdigest()
    return items, digest, lines_digest


def assert_mechanical_diff(old_source: str, new_source: str) -> None:
    anchor = "def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()\n"
    addition = (
        "def canonical_numpy(value):\n"
        " if isinstance(value,np.generic):return value.item()\n"
        " if isinstance(value,np.ndarray):return value.tolist()\n"
        " raise TypeError(f\"Object of type {type(value).__name__} is not JSON serializable\")\n"
    )
    expected = old_source.replace(anchor, anchor + addition, 1).replace(
        'json.dump(obj,f,sort_keys=True,indent=2);',
        'json.dump(obj,f,sort_keys=True,indent=2,default=canonical_numpy);',
        1,
    )
    if expected != new_source:
        raise RuntimeError("new auditor differs beyond the canonical NumPy serializer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--recomputed-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pre_path = args.preregistration.resolve()
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if pre.get("format") != PRE_FORMAT or pre.get("status") != "preregistered_immutable_batch0_audit_reconciliation_authorized":
        raise RuntimeError("wrong v480 preregistration")
    if pre.get("authorization_before_reconciliation") != {
        "phase_b_resume_authorized": False,
        "next_batch_id": None,
        "batch0_reuse_required": True,
        "batch0_rerun_authorized": False,
        "effect_or_outcome_conditioned_retry": False,
        "training_authorized": False,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    }:
        raise RuntimeError("unsafe pre-authorization")
    new_reg = Path(pre["registry_root"]).resolve()
    if args.recomputed_audit.resolve() != Path(pre["outputs"]["recomputed_batch0_audit"]).resolve() or args.output.resolve() != Path(pre["outputs"]["reconciliation_receipt"]).resolve():
        raise RuntimeError("wrong v480 output paths")
    if args.recomputed_audit.parent.resolve() != new_reg or args.output.parent.resolve() != new_reg:
        raise RuntimeError("outputs must remain in the new v480 registry")
    for path in (args.recomputed_audit, args.output):
        if path.exists() or path.with_name(path.name + ".tmp").exists():
            raise FileExistsError(path)

    evidence = {key: require(value) for key, value in pre["immutable_evidence"].items()}
    contract = json.loads(evidence["reconciliation_contract"].read_text(encoding="utf-8"))
    if contract.get("format") != "strict-track2-v480-v478-phase-b-batch0-immutable-reconciliation-contract-v1" or contract.get("status") != "frozen_reconciliation_semantics_execution_preregistration_pending":
        raise RuntimeError("wrong frozen reconciliation contract")
    if contract.get("pass_authorization") != {
        "phase_b_resume_authorized": True,
        "next_batch_id": 1,
        "batch0_reuse_required": True,
        "batch0_rerun_authorized": False,
        "effect_or_outcome_conditioned_retry": False,
        "training_authorized": False,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    } or contract.get("failure_authorization") != {
        "phase_b_resume_authorized": False,
        "batch0_rerun_authorized": False,
        "training_authorized": False,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    }:
        raise RuntimeError("reconciliation contract authorization drift")
    algorithm = contract.get("reconciliation_algorithm", {})
    if algorithm.get("read_only_parent_dataset_and_attempt") is not True or algorithm.get("outputs_only_in_new_registry") is not True or algorithm.get("simulator_or_collector_invocations") != 0 or algorithm.get("parent_evidence_hashes_must_match_before_and_after") is not True:
        raise RuntimeError("reconciliation contract algorithm drift")
    if any(Path(path).exists() for path in pre["required_absence"]):
        raise RuntimeError("legacy formal success artifact unexpectedly exists")
    this_source = Path(__file__).resolve()
    if this_source != Path(pre["execution_closure"]["reconciler"]["path"]).resolve() or sha(this_source) != pre["execution_closure"]["reconciler"]["sha256"]:
        raise RuntimeError("reconciler source closure mismatch")
    old_auditor = evidence["legacy_auditor"]
    new_auditor = evidence["fixed_auditor"]
    assert_mechanical_diff(old_auditor.read_text(encoding="utf-8"), new_auditor.read_text(encoding="utf-8"))
    for source in (this_source, new_auditor):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name.split(".")[0] in {"subprocess", "sapien", "mplib", "toppra"} for name in names):
                    raise RuntimeError("simulator/process import forbidden")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"VectorEnv", "collect_one", "Popen", "run", "call", "check_call", "check_output"}:
                    raise RuntimeError(f"forbidden reconciliation call: {name}")

    batch = Path(pre["batch0_root"]).resolve()
    before, digest_before, lines_before = inventory(batch)
    if not (digest_before == TREE_DIGEST == pre["batch0_tree"]["canonical_compact_json_relative_sha256_pairs_sha256"] and lines_before == TREE_LINES_DIGEST == pre["batch0_tree"]["sorted_sha256_double_space_relative_posix_lines_sha256"] and len(before) == TREE_FILES == pre["batch0_tree"]["file_count"] and sum(int(x[1]) for x in before) == TREE_BYTES == pre["batch0_tree"]["logical_regular_file_bytes"]):
        raise RuntimeError("immutable batch0 tree drift")
    if sha(batch / "batch_report.json") != pre["batch0_tree"]["batch_report_sha256"]:
        raise RuntimeError("batch0 report drift")
    failure = json.loads(evidence["launcher_failure"].read_text(encoding="utf-8"))
    if failure.get("stage") != "audit_batch_0_restore1" or failure.get("exit_code") != 1 or failure.get("v218_health_restored") is not True:
        raise RuntimeError("wrong frozen launcher failure")
    log = evidence["legacy_audit_log"].read_text(encoding="utf-8", errors="replace")
    if "Object of type bool_ is not JSON serializable" not in log:
        raise RuntimeError("wrong frozen serialization failure")

    spec = importlib.util.spec_from_file_location("v478_frozen_batch_auditor", old_auditor)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen parent auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != old_auditor:
        raise RuntimeError("frozen parent auditor import drift")
    parent_pre = json.loads(evidence["phase_b_preregistration"].read_text(encoding="utf-8"))
    a = SimpleNamespace(
        generator=evidence["generator"],
        collector=evidence["collector"],
        selection=evidence["selection"],
        dataset=batch.parent,
        support_root=Path(parent_pre["execution_closure"]["support_root"]["path"]),
        task_config=Path(parent_pre["execution_closure"]["task_config"]["path"]),
        resize_source=Path(parent_pre["execution_closure"]["resize_source"]["path"]),
    )
    result = module.audit_batch(a, parent_pre, 0)
    atomic(args.recomputed_audit, result)
    recomputed = json.loads(args.recomputed_audit.read_text(encoding="utf-8"))
    tmp_bytes = evidence["legacy_audit_tmp"].read_bytes()
    canonical_bytes = args.recomputed_audit.read_bytes()
    tmp_prefix = len(tmp_bytes) < len(canonical_bytes) and canonical_bytes.startswith(tmp_bytes)
    semantic_pass = (
        recomputed.get("format") == BATCH_AUDIT_FORMAT
        and recomputed.get("passed") is True
        and set(recomputed.get("checks", {})) == EXPECTED_CHECKS
        and all(value is True for value in recomputed["checks"].values())
        and recomputed.get("authorization") == {
            "next_batch_resume_authorized": True,
            "technical_effect_used_for_authorization": False,
            "training_authorized": False,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        }
        and recomputed.get("batch_report_sha256") == pre["batch0_tree"]["batch_report_sha256"]
    )
    after, digest_after, lines_after = inventory(batch)
    tree_unchanged = before == after and digest_before == digest_after and lines_before == lines_after
    evidence_unchanged = all(sha(evidence[key]) == record["sha256"] for key, record in pre["immutable_evidence"].items()) and not any(Path(path).exists() for path in pre["required_absence"])
    passed = bool(tmp_prefix and semantic_pass and tree_unchanged and evidence_unchanged)
    receipt = {
        "format": OUT_FORMAT,
        "passed": passed,
        "checks": {
            "legacy_failure_exactly_numpy_bool_serialization": True,
            "fixed_auditor_diff_json_serializer_only": True,
            "legacy_tmp_bitexact_prefix_of_canonical_audit": bool(tmp_prefix),
            "seven_batch0_checks_recomputed_true": bool(semantic_pass),
            "batch0_tree_unchanged": bool(tree_unchanged),
            "all_bound_evidence_unchanged": bool(evidence_unchanged),
            "no_simulator_or_collection_invoked": True,
        },
        "recomputed_batch0_audit": {
            "path": str(args.recomputed_audit.resolve()),
            "sha256": sha(args.recomputed_audit),
            "format": recomputed["format"],
            "checks": recomputed["checks"],
        },
        "batch0_tree": {
            "canonical_compact_json_relative_sha256_pairs_sha256": digest_after,
            "sorted_sha256_double_space_relative_posix_lines_sha256": lines_after,
            "file_count": len(after),
            "logical_regular_file_bytes": sum(int(x[1]) for x in after),
        },
        "semantic_change": {
            "scope": "json_serialization_boundary_only",
            "numpy_generic": "item()",
            "numpy_ndarray": "tolist()",
            "audit_predicates": "bitexact source outside serializer",
        },
        "authorization": {
            "phase_b_resume_authorized": passed,
            "next_batch_id": 1 if passed else None,
            "batch0_reuse_required": True,
            "batch0_rerun_authorized": False,
            "effect_or_outcome_conditioned_retry": False,
            "training_authorized": False,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        },
    }
    atomic(args.output, receipt)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
