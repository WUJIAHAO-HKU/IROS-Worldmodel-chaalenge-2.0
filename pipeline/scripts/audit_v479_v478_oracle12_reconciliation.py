#!/usr/bin/env python3
"""Immutable v479 reconciliation for the v478 endpoint-oracle12 audit.

The frozen v478 auditor is imported and executed without changing its audit
logic.  The only compatibility shim is a canonical JSON encoder for NumPy
values at the final receipt serialization boundary.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


FORMAT = "strict-track2-v479-v478-endpoint-oracle12-immutable-reconciliation-v1"
PRE_FORMAT = "strict-track2-v479-v478-endpoint-oracle12-reconciliation-preregistration-v1"
LEGACY_AUDIT_FORMAT = "strict-track2-v478-endpoint-oracle12-audit-v1"
TREE_DIGEST = "240eb073f89de75100cabfc8fbf278fea2e5cf7fe1f08e016ec306367cede7e0"
TREE_FILE_COUNT = 25
TREE_TOTAL_BYTES = 14034631


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_inventory(root: Path) -> tuple[list[list[object]], str]:
    root = root.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("oracle tree symlink forbidden")
    items = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        items.append([path.relative_to(root).as_posix(), path.stat().st_size, sha(path)])
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return items, digest


def canonical_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, value) -> None:
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


def require_file(record: dict) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or sha(path) != record["sha256"]:
        raise RuntimeError(f"immutable evidence mismatch: {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--legacy-output", type=Path, required=True)
    args = ap.parse_args()

    pre_path = args.preregistration.resolve()
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if pre.get("format") != PRE_FORMAT or pre.get("status") != "preregistered_immutable_reconciliation_authorized":
        raise RuntimeError("wrong reconciliation preregistration")
    if pre["authorization_before_reconciliation"] != {
        "phase_b_temporal_collection_authorized": False,
        "training_authorized": False,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    }:
        raise RuntimeError("unsafe pre-authorization")
    new_reg = Path(pre["registry_root"]).resolve()
    if args.output.resolve() != Path(pre["outputs"]["reconciliation_receipt"]).resolve() or args.legacy_output.resolve() != Path(pre["outputs"]["recomputed_v478_audit"]).resolve():
        raise RuntimeError("outputs are not the preregistered new-registry paths")
    if args.output.parent.resolve() != new_reg or args.legacy_output.parent.resolve() != new_reg:
        raise RuntimeError("outputs must remain inside the v479 registry")
    if args.output.exists() or args.output.with_name(args.output.name + ".tmp").exists() or args.legacy_output.exists() or args.legacy_output.with_name(args.legacy_output.name + ".tmp").exists():
        raise FileExistsError("reconciliation outputs must be absent")

    evidence_paths = {key: require_file(rec) for key, rec in pre["immutable_evidence"].items()}
    if any(Path(path).exists() for path in pre["required_absence"]):
        raise RuntimeError("legacy formal success artifact unexpectedly exists")
    legacy_auditor = evidence_paths["legacy_auditor"]
    this_source = Path(__file__).resolve()
    if this_source != Path(pre["execution_closure"]["reconciler"]["path"]).resolve() or sha(this_source) != pre["execution_closure"]["reconciler"]["sha256"]:
        raise RuntimeError("reconciler source closure mismatch")
    forbidden_names = {"VectorEnv", "collect_one", "Popen", "run", "call", "check_call", "check_output"}
    for source_path in (this_source, legacy_auditor):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name.split(".")[0] in {"subprocess", "sapien", "mplib", "toppra"} for name in names):
                    raise RuntimeError("simulator/process import forbidden in reconciliation")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in forbidden_names:
                    raise RuntimeError(f"forbidden execution call: {name}")
    oracle_root = Path(pre["oracle_root"]).resolve()
    inventory_before, digest_before = tree_inventory(oracle_root)
    if not (
        digest_before == TREE_DIGEST == pre["oracle_tree"]["digest_sha256"]
        and len(inventory_before) == TREE_FILE_COUNT == pre["oracle_tree"]["file_count"]
        and sum(int(x[1]) for x in inventory_before) == TREE_TOTAL_BYTES == pre["oracle_tree"]["total_bytes"]
    ):
        raise RuntimeError("oracle tree drift")
    expected_rows = pre["row_files"]
    for record in expected_rows:
        require_file(record["npz"])
        require_file(record["receipt"])
    if len(expected_rows) != 12:
        raise RuntimeError("expected exact12 immutable rows")

    audit_log = evidence_paths["legacy_auditor_log"].read_text(encoding="utf-8", errors="replace")
    failure = json.loads(evidence_paths["legacy_launcher_failure"].read_text(encoding="utf-8"))
    if "Object of type bool_ is not JSON serializable" not in audit_log:
        raise RuntimeError("legacy failure was not the frozen serialization failure")
    if failure.get("stage") != "endpoint_oracle12_audit" or int(failure.get("exit_code", -1)) == 0:
        raise RuntimeError("legacy launcher failure semantics mismatch")

    spec = importlib.util.spec_from_file_location("v478_frozen_auditor", legacy_auditor)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen auditor")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    if Path(legacy.__file__).resolve() != legacy_auditor:
        raise RuntimeError("legacy auditor import path mismatch")

    legacy_args = pre["legacy_recompute_arguments"]
    argv = [
        str(legacy_auditor),
        "--preregistration", str(evidence_paths["phase_a_preregistration"]),
        "--selection", str(evidence_paths["selection"]),
        "--collector", str(evidence_paths["collector"]),
        "--oracle-root", str(oracle_root),
        "--generation-report", str(evidence_paths["generation_report"]),
        "--support-root", legacy_args["support_root"],
        "--task-config", str(evidence_paths["task_config"]),
        "--resize-source", str(evidence_paths["resize_source"]),
        "--output", str(args.legacy_output.resolve()),
    ]
    original_argv = sys.argv
    original_dump = json.dump

    def canonical_dump(obj, fp, *dump_args, **dump_kwargs):
        if "default" in dump_kwargs:
            raise RuntimeError("unexpected legacy JSON default")
        return original_dump(obj, fp, *dump_args, default=canonical_numpy, **dump_kwargs)

    try:
        # legacy.json is the standard json module object.  This narrowly changes
        # its final serialization boundary while leaving every audit computation
        # and authorization predicate in the frozen module untouched.
        legacy.json.dump = canonical_dump
        sys.argv = argv
        rc = int(legacy.main())
    finally:
        legacy.json.dump = original_dump
        sys.argv = original_argv
    if rc != 0:
        raise RuntimeError(f"frozen auditor recomputation failed: {rc}")

    recomputed = json.loads(args.legacy_output.read_text(encoding="utf-8"))
    legacy_tmp = evidence_paths["legacy_audit_tmp"].read_bytes()
    recomputed_bytes = args.legacy_output.read_bytes()
    legacy_tmp_exact_prefix = len(legacy_tmp) < len(recomputed_bytes) and recomputed_bytes.startswith(legacy_tmp)
    checks = recomputed.get("checks", {})
    expected_authorization = {
        "phase_b_temporal_collection_authorized": True,
        "training_authorized": False,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    }
    semantic_pass = (
        recomputed.get("format") == LEGACY_AUDIT_FORMAT
        and recomputed.get("passed") is True
        and checks
        and all(value is True for value in checks.values())
        and recomputed.get("authorization") == expected_authorization
    )
    inventory_after, digest_after = tree_inventory(oracle_root)
    immutable_tree = inventory_after == inventory_before and digest_after == digest_before
    old_evidence_unchanged = all(
        sha(evidence_paths[key]) == record["sha256"]
        for key, record in pre["immutable_evidence"].items()
    ) and all(
        sha(Path(record[k]["path"])) == record[k]["sha256"]
        for record in expected_rows for k in ("npz", "receipt")
    ) and not any(Path(path).exists() for path in pre["required_absence"])
    passed = bool(semantic_pass and immutable_tree and old_evidence_unchanged and legacy_tmp_exact_prefix)
    receipt = {
        "format": FORMAT,
        "passed": passed,
        "checks": {
            "all_frozen_v478_checks_recomputed_true": bool(semantic_pass),
            "legacy_failure_exactly_numpy_bool_serialization": True,
            "oracle_tree_unchanged": bool(immutable_tree),
            "all_bound_old_evidence_unchanged": bool(old_evidence_unchanged),
            "legacy_tmp_is_bitexact_prefix_of_canonical_receipt": bool(legacy_tmp_exact_prefix),
            "no_simulator_or_collection_invoked": True,
            "semantic_diff_serialization_only": bool(legacy_tmp_exact_prefix),
        },
        "recomputed_v478_audit": {
            "path": str(args.legacy_output.resolve()),
            "sha256": sha(args.legacy_output),
            "format": recomputed["format"],
            "checks": checks,
        },
        "immutable_evidence_sha256": {
            key: record["sha256"] for key, record in pre["immutable_evidence"].items()
        },
        "oracle_tree": {
            "digest_sha256": digest_after,
            "file_count": len(inventory_after),
            "total_bytes": sum(int(x[1]) for x in inventory_after),
        },
        "semantic_change": {
            "scope": "json_serialization_boundary_only",
            "numpy_generic": "item()",
            "numpy_ndarray": "tolist()",
            "audit_computation": "frozen v478 auditor main() invoked unchanged",
        },
        "authorization": {
            "phase_b_temporal_collection_authorized": passed,
            "training_authorized": False,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        },
    }
    atomic_json(args.output, receipt)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
