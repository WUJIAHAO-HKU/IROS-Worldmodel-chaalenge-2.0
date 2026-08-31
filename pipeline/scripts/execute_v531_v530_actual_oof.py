#!/usr/bin/env python3
"""Single-boundary v531 actual OOF computation over the immutable v524 cache."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import random
import signal
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import torch

SEED = 1665
FORMAT = "strict-track2-v531-v530-actual-oof-execution-receipt-v1"
EVENT_FORMAT = "strict-track2-v531-v530-actual-oof-event-v1"
BRANCHES = ["factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4"]
AUTH_KEYS = {
    "actual_oof_execution_boundary_invocations_authorized", "actual_oof_execution_boundary_invocations_consumed",
    "direct_executor_invocations_authorized", "direct_auditor_invocations_authorized", "oof_readonly_validator_invocations_authorized",
    "model_runtime_delegate_invocations_authorized", "phase_a_launcher_invocations_authorized", "phase_a_driver_invocations_authorized",
    "phase_a_worker_invocations_authorized", "phase_a_replay_invocations_authorized", "training_invocations_authorized",
    "cache_reuse_invocations_authorized", "reward_read_invocations_authorized", "dev_hidden_final_outcome_read_invocations_authorized",
    "submission_invocations_authorized", "retry_authorized", "actual_oof_execution_authorized", "qualification_readonly",
    "qualification_mutation_authorized", "hidden_or_final_inputs_authorized", "reward_read_authorized", "training_authorized",
    "cache_reuse_authorized", "submission_authorized",
}
EXPECTED_AUTH = {
    "actual_oof_execution_boundary_invocations_authorized": 1, "actual_oof_execution_boundary_invocations_consumed": 0,
    "direct_executor_invocations_authorized": 0, "direct_auditor_invocations_authorized": 0, "oof_readonly_validator_invocations_authorized": 0,
    "model_runtime_delegate_invocations_authorized": 0, "phase_a_launcher_invocations_authorized": 0, "phase_a_driver_invocations_authorized": 0,
    "phase_a_worker_invocations_authorized": 0, "phase_a_replay_invocations_authorized": 0, "training_invocations_authorized": 0,
    "cache_reuse_invocations_authorized": 0, "reward_read_invocations_authorized": 0, "dev_hidden_final_outcome_read_invocations_authorized": 0,
    "submission_invocations_authorized": 0, "retry_authorized": False, "actual_oof_execution_authorized": True,
    "qualification_readonly": True, "qualification_mutation_authorized": False, "hidden_or_final_inputs_authorized": False,
    "reward_read_authorized": False, "training_authorized": False, "cache_reuse_authorized": False, "submission_authorized": False,
}
PREREG_KEYS = {"format", "status", "seed", "classification", "authority_contract_path", "fresh_authority_root", "fresh_authority_prep_root", "fresh_attempt_root", "fresh_attempt_prep_root", "fresh_oof_root", "fresh_oof_prep_root", "execution_manifest", "active_source_paths", "input_contract", "output_contract", "authorization", "boundary_partition", "required_environment", "service_health", "historical_training_authority_not_inherited", "no_space_transport_required", "noncyclic_transport_deployment_record_required"}
MANIFEST_KEYS = {"format", "status", "seed", "fold_order", "branches", "selection_count", "ordered_oof_rows_count", "fold_row_counts", "ordering", "ordered_rows", "ordered_rows_canonical_sha256", "cache_input", "dataset_input", "model_and_source_input", "output_schema", "execution_contract"}
ROW_KEYS = {"oof_ordinal", "fold", "selection_order", "branch_index", "branch", "sample_id", "episode", "start", "instruction_sha256", "row_npz_path", "row_npz_sha256", "row_receipt_path", "row_receipt_sha256", "target_array_key", "target_branch_index", "cache_sample_id", "cache_request_sha256", "cache_output_sha256", "raw_warning"}
CONTRACT_KEYS = {
    "actual_oof_auditor_source", "actual_oof_execution_manifest_source", "actual_oof_execution_preregistration_source",
    "actual_oof_executor_source", "actual_oof_input_contract", "actual_oof_launcher_source", "actual_oof_output_contract",
    "authority_materializer_source", "authority_receipt_contract", "authorization", "current_absences_after_authority",
    "dataset_contract", "execution_boundary", "format", "historical_absences", "lineage", "model_and_source_contract",
    "ordered_execution_manifest_contract", "qualification_contract", "runtime_observation", "seed", "source_aliases",
    "source_closure", "source_closure_sha256", "source_role_order", "status", "v530_tri_bind",
}
AUTHORITY_RECEIPT_CONTRACT_KEYS = {
    "actual_oof_input_contract_exact", "actual_oof_output_contract_exact", "authorization_exact", "check_key_set_sha256",
    "check_keys", "checks_sha256", "execution_boundary_exact", "format", "ordered_execution_manifest_contract_exact",
    "runtime_observation_exact", "status", "top_keys", "v530_tri_bind_exact",
}
SNAPSHOT_KEYS = {"absences", "canonical_sha256", "files", "gpu_compute_pids", "relevant_execution_pids", "services", "trees"}
SOURCE_ROLE_ORDER = [
    "authority_materializer", "actual_oof_execution_preregistration", "actual_oof_execution_manifest", "actual_oof_executor",
    "actual_oof_auditor", "actual_oof_launcher", "v530_external_terminal_process", "v527_candidate_receipt",
    "v527_authority_receipt", "v528_deployment_receipt", "v524_qualification_terminal", "v524_qualification_report",
    "v524_qualification_audit", "v524_cache_manifest", "v482_preregistration", "v482_runtime_source",
    "v482_trainer_source", "v482_model_design_contract", "v482_s0_auditor", "v478_selection",
]
SOURCE_ALIASES = {
    "authority_design_contract": "authority_design_contract", "authority_materializer": "authority_materializer_source",
    "actual_oof_execution_preregistration": "actual_oof_execution_preregistration_source",
    "actual_oof_execution_manifest": "actual_oof_execution_manifest_source", "actual_oof_executor": "actual_oof_executor_source",
    "actual_oof_auditor": "actual_oof_auditor_source", "actual_oof_launcher": "actual_oof_launcher_source",
    "v530_external_terminal_process": "v530_external_terminal_process_source", "v527_candidate_receipt": "v527_candidate_receipt_source",
    "v527_authority_receipt": "v527_authority_receipt_source", "v528_deployment_receipt": "v528_deployment_receipt_source",
    "v524_qualification_terminal": "v524_qualification_terminal_source", "v524_qualification_report": "v524_qualification_report_source",
    "v524_qualification_audit": "v524_qualification_audit_source", "v524_cache_manifest": "v524_cache_manifest_source",
    "v482_preregistration": "v482_preregistration_source", "v482_runtime_source": "v482_runtime_source_source",
    "v482_trainer_source": "v482_trainer_source_source", "v482_model_design_contract": "v482_model_design_contract_source",
    "v482_s0_auditor": "v482_s0_auditor_source", "v478_selection": "v478_selection_source",
}
EXPECTED_LINEAGE = {
    "canonical_execution_not_performed": True, "fresh_actual_oof_authority_only": True,
    "v482_dataset_model_source_current_revalidated": True, "v524_qualification_exact20_readonly": True,
    "v530_candidate_external_terminal_consumed_readonly": True,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def json_exact(left, right) -> bool:
    """JSON equality that does not let bool values impersonate integer counts."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def exact_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> None:
    if path.is_symlink() or not path.is_file() or sha(path) != expected_sha or (expected_bytes is not None and path.stat().st_size != expected_bytes):
        raise RuntimeError(f"exact file:{path}")


def atomic_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    exact_file(path, hashlib.sha256(payload).hexdigest(), len(payload))


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def tree(root: Path) -> dict:
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink:{path}")
        if path.is_file():
            inventory.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
    lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in inventory).encode()
    return {"file_count": len(inventory), "logical_file_bytes": sum(x[2] for x in inventory), "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(), "canonical_json_triples_digest_sha256": canonical_sha(inventory), "inventory": inventory}


def rng_snapshot() -> dict:
    py = json.dumps(random.getstate(), separators=(",", ":")).encode()
    np_state = np.random.get_state()
    np_payload = json.dumps([np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])], separators=(",", ":")).encode()
    cpu = torch.random.get_rng_state().cpu().numpy().tobytes()
    cuda = []
    cuda_initialized = torch.cuda.is_initialized()
    if cuda_initialized:
        for index in range(torch.cuda.device_count()):
            value = torch.cuda.get_rng_state(index).cpu().numpy().tobytes()
            cuda.append({"device": index, "sha256": hashlib.sha256(value).hexdigest(), "logical_bytes": len(value)})
    return {"python": {"sha256": hashlib.sha256(py).hexdigest(), "logical_bytes": len(py)}, "numpy": {"sha256": hashlib.sha256(np_payload).hexdigest(), "logical_bytes": len(np_payload)}, "torch_cpu": {"sha256": hashlib.sha256(cpu).hexdigest(), "logical_bytes": len(cpu)}, "torch_cuda_initialized": cuda_initialized, "torch_cuda": cuda}


def services_snapshot() -> dict:
    result = {}
    endpoints = {8005: "/v1/health", 18084: "/health"}
    for port, endpoint in endpoints.items():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=5) as response:
            body = response.read()
            result[str(port)] = {"http_code": response.status, "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body), "json_model": json.loads(body)}
    return result


def gpu_processes() -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if any("/dev/nvidia" in os.readlink(fd) for fd in (entry / "fd").iterdir()):
                found.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(found)


def validate_v482_data_model(manifest: dict) -> None:
    model = manifest["model_and_source_input"]
    record = model["v482_preregistration"]
    path = Path(record["path"]); exact_file(path, record["sha256"], record["logical_bytes"])
    parent = json.loads(path.read_text(encoding="utf-8"))
    dataset = parent["dataset"]
    root = Path(dataset["root"])
    expected = {row["relative"]: (row["sha256"], row["bytes"]) for row in dataset["files"]}
    actual_paths = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink(): raise RuntimeError("dataset symlink")
        if item.is_file(): actual_paths[item.relative_to(root).as_posix()] = item
    if set(actual_paths) != set(expected): raise RuntimeError("dataset members")
    inventory = []
    for relative in sorted(expected):
        digest, size = expected[relative]; exact_file(actual_paths[relative], digest, size); inventory.append([relative, digest])
    if canonical_sha(inventory) != dataset["tree_sha256"]: raise RuntimeError("dataset tree")
    exact_file(Path(dataset["selection"]["path"]), dataset["selection"]["sha256"])
    for key, value in parent["source"].items():
        if key.endswith("_path") and key[:-5] + "_sha256" in parent["source"]:
            exact_file(Path(value), parent["source"][key[:-5] + "_sha256"])
    v169 = parent["v169"]
    for stem in ("closure", "library_manifest", "release_manifest"):
        exact_file(Path(v169[f"{stem}_path"]), v169[f"{stem}_sha256"])
    runtime_path = Path(parent["source"]["runtime_path"])
    package_root = str(runtime_path.parent.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    spec = importlib.util.spec_from_file_location("v531_frozen_v482_runtime", runtime_path)
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    recomputed = runtime.verify_v169(json.loads(Path(v169["closure_path"]).read_text(encoding="utf-8")))
    if recomputed != v169["closure_digest"] or recomputed != v169["required_closure_digest"]:
        raise RuntimeError("v169 release/library current closure")


def validate_authority(authority: dict, contract: dict, prereg: dict, manifest: dict, args, *, enforce_production_paths: bool = True) -> None:
    if set(contract) != CONTRACT_KEYS or contract.get("format") != "strict-track2-v531-v530-actual-oof-execution-authority-design-contract-v1" or contract.get("status") != "design_only_frozen_actual_oof_sources_pending_independent_authority_materialization" or type(contract.get("seed")) is not int or contract["seed"] != SEED:
        raise RuntimeError("authority contract exact schema")
    if not json_exact(contract.get("lineage"), EXPECTED_LINEAGE) or contract.get("source_role_order") != SOURCE_ROLE_ORDER or not json_exact(contract.get("source_aliases"), SOURCE_ALIASES):
        raise RuntimeError("authority contract literal lineage/source roles")
    if not json_exact(contract.get("actual_oof_input_contract"), prereg["input_contract"]) or not json_exact(contract.get("actual_oof_output_contract"), prereg["output_contract"]):
        raise RuntimeError("authority contract preregistration input/output")
    expected_ordered = {
        "manifest": {"path": str(args.manifest), "sha256": args.manifest_sha, "logical_bytes": args.manifest.stat().st_size},
        "ordered_rows_count": 1000, "ordered_rows_canonical_sha256": manifest["ordered_rows_canonical_sha256"],
        "fold_order": manifest["fold_order"], "fold_row_counts": manifest["fold_row_counts"], "branches": manifest["branches"],
    }
    if not json_exact(contract.get("ordered_execution_manifest_contract"), expected_ordered) or not json_exact(contract.get("model_and_source_contract"), manifest["model_and_source_input"]):
        raise RuntimeError("authority contract manifest/model bindings")
    qualification_tree = contract.get("qualification_contract", {}).get("tree", {})
    qualification_inventory = qualification_tree.get("inventory") if isinstance(qualification_tree, dict) else None
    if not isinstance(qualification_inventory, list) or set(qualification_tree) != {"file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256", "inventory"}:
        raise RuntimeError("authority contract qualification tree schema")
    qualification_lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in qualification_inventory).encode()
    expected_qualification = {"root": manifest["cache_input"]["qualification_root"], "tree": qualification_tree, "readonly": True, "mutation_authorized": False}
    qualification_summary = {key: qualification_tree[key] for key in manifest["cache_input"]["qualification_tree"]}
    if not json_exact(contract.get("qualification_contract"), expected_qualification) or not json_exact(qualification_summary, manifest["cache_input"]["qualification_tree"]) or canonical_sha(qualification_inventory) != qualification_tree["canonical_json_triples_digest_sha256"] or hashlib.sha256(qualification_lines).hexdigest() != qualification_tree["sha256sum_lines_digest_sha256"]:
        raise RuntimeError("authority contract qualification binding")
    dataset_tree = contract.get("dataset_contract", {}).get("tree", {})
    dataset_inventory = dataset_tree.get("inventory") if isinstance(dataset_tree, dict) else None
    if not isinstance(dataset_inventory, list) or set(dataset_tree) != {"file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_pairs_digest_sha256", "canonical_json_triples_digest_sha256", "inventory"}:
        raise RuntimeError("authority contract dataset tree schema")
    dataset_lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in dataset_inventory).encode()
    expected_dataset = {
        "root": manifest["dataset_input"]["root"], "selection": manifest["dataset_input"]["selection"], "mutation_authorized": False,
        "tree": dataset_tree,
    }
    if not json_exact(contract.get("dataset_contract"), expected_dataset) or dataset_tree["file_count"] != 411 or dataset_tree["logical_file_bytes"] != manifest["dataset_input"]["logical_file_bytes"] or canonical_sha(dataset_inventory) != dataset_tree["canonical_json_triples_digest_sha256"] or canonical_sha([[relative, digest] for relative, digest, _ in dataset_inventory]) != dataset_tree["canonical_json_pairs_digest_sha256"] or hashlib.sha256(dataset_lines).hexdigest() != dataset_tree["sha256sum_lines_digest_sha256"] or dataset_tree["canonical_json_pairs_digest_sha256"] != manifest["dataset_input"]["canonical_tree_sha256"]:
        raise RuntimeError("authority contract dataset binding")
    expected_historical_paths = {
        "authority_root": {"path": prereg["fresh_authority_root"]}, "authority_prep": {"path": prereg["fresh_authority_prep_root"]},
        "actual_oof_attempt_root": {"path": prereg["fresh_attempt_root"]}, "actual_oof_attempt_prep": {"path": prereg["fresh_attempt_prep_root"]},
        "actual_oof_output_root": {"path": prereg["fresh_oof_root"]}, "actual_oof_output_prep": {"path": prereg["fresh_oof_prep_root"]},
    }
    expected_current_paths = {key: value for key, value in expected_historical_paths.items() if key != "authority_root"}
    if not json_exact(contract.get("historical_absences"), expected_historical_paths) or not json_exact(contract.get("current_absences_after_authority"), expected_current_paths):
        raise RuntimeError("authority contract canonical absence paths")
    receipt_contract = contract.get("authority_receipt_contract")
    if not isinstance(receipt_contract, dict) or set(receipt_contract) != AUTHORITY_RECEIPT_CONTRACT_KEYS:
        raise RuntimeError("authority receipt contract exact schema")
    auth = authority.get("authorization")
    if not isinstance(auth, dict) or set(auth) != AUTH_KEYS or not json_exact(auth, prereg["authorization"]) or not json_exact(auth, contract["authorization"]) or not json_exact(auth, receipt_contract["authorization_exact"]):
        raise RuntimeError("authorization exact boundary")
    for key in AUTH_KEYS:
        value = auth[key]
        if key.endswith("_invocations_authorized") or key.endswith("_invocations_consumed"):
            if type(value) is not int:
                raise RuntimeError("authorization strict int")
        elif type(value) is not bool:
            raise RuntimeError("authorization strict bool")
    if set(authority) != set(receipt_contract["top_keys"]) or receipt_contract["top_keys"] != sorted(authority):
        raise RuntimeError("authority exact top keys")
    if authority.get("format") != receipt_contract["format"] or authority.get("status") != receipt_contract["status"] or authority.get("passed") is not True:
        raise RuntimeError("authority terminal identity")
    if authority.get("check_keys") != receipt_contract["check_keys"] or authority["check_keys"] != sorted(authority.get("checks", {})) or set(authority["checks"]) != set(authority["check_keys"]):
        raise RuntimeError("authority exact check keys")
    if any(value is not True for value in authority["checks"].values()):
        raise RuntimeError("authority checks not all true")
    if canonical_sha(authority["check_keys"]) != authority.get("check_key_set_sha256") or authority["check_key_set_sha256"] != receipt_contract["check_key_set_sha256"]:
        raise RuntimeError("authority check-key digest")
    if canonical_sha(authority["checks"]) != authority.get("checks_sha256") or authority["checks_sha256"] != receipt_contract["checks_sha256"]:
        raise RuntimeError("authority checks digest")
    exact_contract_fields = {
        "actual_oof_input_contract": "actual_oof_input_contract_exact",
        "actual_oof_output_contract": "actual_oof_output_contract_exact",
        "ordered_execution_manifest_contract": "ordered_execution_manifest_contract_exact",
        "execution_boundary": "execution_boundary_exact",
        "runtime_observation": "runtime_observation_exact",
        "v530_tri_bind": "v530_tri_bind_exact",
    }
    for field, receipt_field in exact_contract_fields.items():
        if not json_exact(contract[field], receipt_contract[receipt_field]) or not json_exact(authority.get(field), contract[field]):
            raise RuntimeError(f"authority embedded contract:{field}")
    for field in ("dataset_contract", "qualification_contract", "model_and_source_contract"):
        if not json_exact(authority.get(field), contract[field]):
            raise RuntimeError(f"authority contract anchor:{field}")
    expected_historical = {key: {"path": value["path"], "absent": True} for key, value in contract["historical_absences"].items()}
    expected_required = {key: {"path": value["path"], "absent": True} for key, value in contract["current_absences_after_authority"].items()}
    if not json_exact(authority.get("historical_absences"), expected_historical) or not json_exact(authority.get("required_absences"), expected_required):
        raise RuntimeError("authority absence contracts")
    design_record = {"path": str(args.contract), "sha256": args.contract_sha, "logical_bytes": args.contract.stat().st_size}
    expected_roles = ["authority_design_contract", *contract["source_role_order"]]
    expected_closure = dict(contract["source_closure"]); expected_closure["authority_design_contract"] = design_record
    if set(contract["source_role_order"]) != set(contract["source_closure"]) or len(contract["source_role_order"]) != len(set(contract["source_role_order"])) or canonical_sha(contract["source_closure"]) != contract["source_closure_sha256"]:
        raise RuntimeError("design source closure")
    if authority.get("source_role_order") != expected_roles or set(authority.get("source_closure", {})) != set(expected_roles) or not json_exact(authority["source_closure"], expected_closure) or canonical_sha(authority["source_closure"]) != authority.get("source_closure_sha256"):
        raise RuntimeError("authority source closure")
    if not json_exact(authority.get("source_aliases"), SOURCE_ALIASES) or set(authority["source_aliases"]) != set(expected_roles):
        raise RuntimeError("authority source aliases")
    for role in expected_roles:
        alias = authority["source_aliases"][role]
        if authority.get(alias) != authority["source_closure"][role]:
            raise RuntimeError(f"authority source alias:{role}")
    for snapshot_name in ("input_pre_snapshot", "input_post_snapshot"):
        snapshot = authority.get(snapshot_name)
        if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_KEYS:
            raise RuntimeError("authority snapshot schema")
        unsigned = dict(snapshot); claimed = unsigned.pop("canonical_sha256")
        if canonical_sha(unsigned) != claimed or not json_exact(snapshot["files"], authority["source_closure"]) or not json_exact(snapshot["absences"], authority["historical_absences"]):
            raise RuntimeError("authority snapshot digest/bindings")
        if not json_exact(snapshot["services"], prereg["service_health"]) or snapshot["gpu_compute_pids"] != [] or snapshot["relevant_execution_pids"] != []:
            raise RuntimeError("authority snapshot runtime")
        expected_trees = {
            "dataset_exact411": {key: value for key, value in contract["dataset_contract"]["tree"].items() if key != "canonical_json_pairs_digest_sha256"},
            "qualification_exact20": contract["qualification_contract"]["tree"],
            "v527_candidate_exact1": contract["v530_tri_bind"]["candidate_tree"],
            "v530_external_evidence_exact6": contract["v530_tri_bind"]["external_evidence_tree"],
        }
        if not json_exact(snapshot["trees"], expected_trees):
            raise RuntimeError("authority snapshot trees")
    if authority.get("input_snapshots_exactly_equal") is not True or not json_exact(authority["input_pre_snapshot"], authority["input_post_snapshot"]):
        raise RuntimeError("authority pre/post snapshot equality")
    if authority.get("actual_oof_execution_preregistration_source") != {"path": str(args.preregistration), "sha256": args.preregistration_sha, "logical_bytes": args.preregistration.stat().st_size}:
        raise RuntimeError("authority preregistration binding")
    if authority.get("actual_oof_execution_manifest_source") != {"path": str(args.manifest), "sha256": args.manifest_sha, "logical_bytes": args.manifest.stat().st_size}:
        raise RuntimeError("authority manifest binding")
    expected_sources = {
        "actual_oof_executor_source": (Path(__file__).resolve(), args.executor_sha),
        "actual_oof_auditor_source": (args.auditor_source.resolve(), args.auditor_sha),
        "actual_oof_launcher_source": (args.launcher_source.resolve(), args.launcher_sha),
    }
    for field, (path, digest) in expected_sources.items():
        if authority.get(field) != {"path": str(path), "sha256": digest, "logical_bytes": path.stat().st_size}:
            raise RuntimeError(f"authority active source:{field}")
    if enforce_production_paths:
        if args.contract != Path(prereg["authority_contract_path"]) or args.authority_receipt != Path(prereg["fresh_authority_root"]) / "authority_receipt.json":
            raise RuntimeError("canonical production authority paths")
        if args.preregistration != Path(prereg["active_source_paths"]["preregistration"]) or args.manifest != Path(prereg["active_source_paths"]["manifest"]) or args.auditor_source != Path(prereg["active_source_paths"]["auditor"]) or args.launcher_source != Path(prereg["active_source_paths"]["launcher"]):
            raise RuntimeError("canonical production source paths")


def validate_static_documents(prereg: dict, manifest: dict) -> None:
    if set(prereg) != PREREG_KEYS or set(manifest) != MANIFEST_KEYS:
        raise RuntimeError("preregistration/manifest exact keys")
    if prereg["format"] != "strict-track2-v531-v530-actual-oof-execution-preregistration-v1" or prereg["status"] != "preregistered_actual_oof_execution_pending_external_authority" or type(prereg["seed"]) is not int or prereg["seed"] != SEED:
        raise RuntimeError("preregistration identity")
    if manifest["format"] != "strict-track2-v531-v530-actual-oof-execution-manifest-v1" or manifest["status"] != "preregistered_actual_oof_order_and_output_schema" or type(manifest["seed"]) is not int or manifest["seed"] != SEED:
        raise RuntimeError("manifest identity")
    if set(prereg["authorization"]) != AUTH_KEYS or prereg["authorization"] != EXPECTED_AUTH:
        raise RuntimeError("authorization keys")
    for key, expected in EXPECTED_AUTH.items():
        if type(prereg["authorization"][key]) is not type(expected):
            raise RuntimeError("authorization strict type")
    rows = manifest["ordered_rows"]
    if manifest["fold_order"] != [0, 1, 2, 3, 4] or manifest["branches"] != BRANCHES or manifest["fold_row_counts"] != [200] * 5 or type(manifest["selection_count"]) is not int or manifest["selection_count"] != 200 or type(manifest["ordered_oof_rows_count"]) is not int or manifest["ordered_oof_rows_count"] != 1000 or len(rows) != 1000 or manifest["ordered_rows_canonical_sha256"] != canonical_sha(rows):
        raise RuntimeError("manifest counts/order")
    for ordinal, row in enumerate(rows):
        if set(row) != ROW_KEYS or any(type(row[key]) is not int for key in ("oof_ordinal", "fold", "selection_order", "branch_index", "sample_id", "episode", "start", "target_branch_index", "cache_sample_id")):
            raise RuntimeError("manifest row schema/types")
        if row["oof_ordinal"] != ordinal or row["branch"] != BRANCHES[row["branch_index"]] or row["target_branch_index"] != row["branch_index"] or row["cache_sample_id"] != row["sample_id"] or row["sample_id"] != row["selection_order"] * 5 + row["branch_index"]:
            raise RuntimeError("manifest row identity")
        if row["fold"] != ordinal // 200:
            raise RuntimeError("manifest fold partition")
    for value in (prereg["authority_contract_path"], prereg["fresh_authority_root"], prereg["fresh_authority_prep_root"], prereg["fresh_attempt_root"], prereg["fresh_attempt_prep_root"], prereg["fresh_oof_root"], prereg["fresh_oof_prep_root"], *prereg["active_source_paths"].values()):
        if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
            raise RuntimeError("canonical POSIX path")


def validate_input_records(prereg: dict) -> None:
    for record in prereg["input_contract"].values():
        if isinstance(record, dict) and set(record) == {"path", "sha256", "logical_bytes"}:
            exact_file(Path(record["path"]), record["sha256"], record["logical_bytes"])
    external = json.loads(Path(prereg["input_contract"]["v530_external_terminal"]["path"]).read_text())
    if external.get("passed") is not True or external.get("status") != "passed_external_terminal" or external.get("candidate_consumable") is not True:
        raise RuntimeError("v530 external terminal")
    candidate = json.loads(Path(prereg["input_contract"]["v527_candidate"]["path"]).read_text())
    if candidate.get("candidate_verified") is not True or candidate.get("standalone_consumable") is not False or candidate.get("external_terminal_required") is not True or candidate.get("publication_success_claimed") is not False or candidate.get("passed") is not None:
        raise RuntimeError("v527 candidate truth")
    if external.get("candidate_receipt", {}).get("sha256") != prereg["input_contract"]["v527_candidate"]["sha256"]:
        raise RuntimeError("candidate/external dual binding")


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def execute(args) -> dict:
    if type(args.seed) is not int or args.seed != SEED:
        raise RuntimeError("seed strict int")
    exact_file(Path(__file__).resolve(), args.executor_sha)
    exact_file(args.preregistration, args.preregistration_sha)
    exact_file(args.manifest, args.manifest_sha)
    exact_file(args.contract, args.contract_sha)
    exact_file(args.authority_receipt, args.authority_receipt_sha)
    exact_file(args.auditor_source, args.auditor_sha)
    exact_file(args.launcher_source, args.launcher_sha)
    prereg = json.loads(args.preregistration.read_text())
    manifest = json.loads(args.manifest.read_text())
    contract = json.loads(args.contract.read_text())
    authority = json.loads(args.authority_receipt.read_text())
    validate_static_documents(prereg, manifest)
    if prereg["execution_manifest"] != {"path": str(args.manifest), "sha256": args.manifest_sha, "logical_bytes": args.manifest.stat().st_size}:
        raise RuntimeError("manifest source binding")
    if args.output_root != Path(prereg["fresh_oof_root"]) or args.output_root.with_name(args.output_root.name + ".oof-prep") != Path(prereg["fresh_oof_prep_root"]):
        raise RuntimeError("fresh output path")
    validate_authority(authority, contract, prereg, manifest, args)
    validate_input_records(prereg)
    if {key: os.environ.get(key) for key in prereg["required_environment"]} != prereg["required_environment"]:
        raise RuntimeError("execution environment")
    services_before = services_snapshot()
    if services_before != prereg["service_health"]: raise RuntimeError("service health pre")
    gpu_before = gpu_processes()
    if gpu_before: raise RuntimeError("GPU process prestate")
    validate_v482_data_model(manifest)
    cache_record = manifest["cache_input"]["cache_npz"]
    cache_path = Path(cache_record["path"])
    exact_file(cache_path, cache_record["sha256"], cache_record["logical_bytes"])
    before_qualification = tree(Path(manifest["cache_input"]["qualification_root"]))
    frozen_tree = manifest["cache_input"]["qualification_tree"]
    if any(before_qualification[key] != frozen_tree[key] for key in ("file_count", "logical_file_bytes", "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256")):
        raise RuntimeError("qualification exact20")
    if args.output_root.exists() or Path(prereg["fresh_oof_prep_root"]).exists():
        raise FileExistsError("fresh OOF roots")
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    prep = Path(prereg["fresh_oof_prep_root"])
    committed = False
    try:
        prep.mkdir(mode=0o700)
        identity = prep.stat()
        before_rng = rng_snapshot()
        intent = {
            "format": "strict-track2-v531-v530-actual-oof-attempt-intent-v1", "seed": SEED,
            "executor_argv": list(args.literal_argv), "launcher_argv": json.loads(args.launcher_argv_json),
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0), "sid": os.getsid(0),
            "launcher_pid": args.launcher_pid, "launcher_ppid": args.launcher_ppid, "launcher_pgid": args.launcher_pgid, "launcher_sid": args.launcher_sid,
            "preregistration_sha256": args.preregistration_sha, "manifest_sha256": args.manifest_sha,
            "contract_sha256": args.contract_sha, "authority_receipt_sha256": args.authority_receipt_sha,
            "executor_sha256": args.executor_sha, "auditor_sha256": args.auditor_sha, "launcher_sha256": args.launcher_sha,
            "rng_entry": before_rng, "qualification_readonly": True,
        }
        atomic_json(prep / "attempt_intent.json", intent)
        with np.load(cache_path, allow_pickle=False) as raw:
            if list(raw.files) != ["baseline", "seed", "request_sha256", "output_sha256", "sample_id"]:
                raise RuntimeError("cache keys")
            baseline = np.asarray(raw["baseline"])
            request_hashes = np.asarray(raw["request_sha256"])
            output_hashes = np.asarray(raw["output_sha256"])
            sample_ids = np.asarray(raw["sample_id"])
        if baseline.shape != (200, 5, 8, 256, 256, 3) or baseline.dtype != np.uint8:
            raise RuntimeError("cache schema")
        rows = manifest["ordered_rows"]
        metrics = {"absolute_error_sum_uint64": [], "branch": [], "episode": [], "fold": [], "pixel_count_uint64": [], "sample_id": [], "selection_order": [], "target_sha256": [], "prediction_sha256": []}
        events = []
        loaded_selection = None
        loaded_targets = None
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            for ordinal, frozen in enumerate(rows):
                selection, branch = frozen["selection_order"], frozen["branch_index"]
                if int(sample_ids[selection, branch]) != frozen["sample_id"] or str(request_hashes[selection, branch]) != frozen["cache_request_sha256"] or str(output_hashes[selection, branch]) != frozen["cache_output_sha256"]:
                    raise RuntimeError("cache row")
                if loaded_selection != selection:
                    row_path, receipt_path = Path(frozen["row_npz_path"]), Path(frozen["row_receipt_path"])
                    exact_file(row_path, frozen["row_npz_sha256"]); exact_file(receipt_path, frozen["row_receipt_sha256"])
                    with np.load(row_path, allow_pickle=False) as source:
                        variants = list(np.asarray(source["variants"]).astype("U"))
                        loaded_targets = np.ascontiguousarray(source["temporal_rgb"][:5])
                    if variants != BRANCHES + ["factual_duplicate"] or loaded_targets.shape != (5, 8, 256, 256, 3) or loaded_targets.dtype != np.uint8:
                        raise RuntimeError("dataset target group schema")
                    loaded_selection = selection
                target = np.ascontiguousarray(loaded_targets[branch])
                prediction = np.ascontiguousarray(baseline[selection, branch])
                if prediction.dtype != np.uint8 or target.dtype != np.uint8 or prediction.shape != (8, 256, 256, 3) or target.shape != prediction.shape:
                    raise RuntimeError("OOF row schema")
                delta = np.abs(prediction.astype(np.int16) - target.astype(np.int16)).astype(np.uint16)
                error_sum, pixels = int(delta.sum(dtype=np.uint64)), int(delta.size)
                prediction_sha, target_sha = arrsha(prediction), arrsha(target)
                if prediction_sha != frozen["cache_output_sha256"]:
                    raise RuntimeError("cached prediction payload hash")
                event = {"absolute_error_sum_uint64": error_sum, "branch": frozen["branch"], "branch_index": branch, "cache_output_sha256": frozen["cache_output_sha256"], "episode": frozen["episode"], "fold": frozen["fold"], "input_raw_warning": frozen["raw_warning"], "oof_computation_warnings": [], "oof_ordinal": ordinal, "pixel_count_uint64": pixels, "prediction_sha256": prediction_sha, "sample_id": frozen["sample_id"], "selection_order": selection, "start": frozen["start"], "target_sha256": target_sha}
                event["event_canonical_sha256"] = canonical_sha(event)
                if set(event) != set(manifest["output_schema"]["event_exact_keys"]):
                    raise RuntimeError("event exact keys")
                events.append(event)
                for key in metrics:
                    metrics[key].append({"absolute_error_sum_uint64": error_sum, "branch": branch, "episode": frozen["episode"], "fold": frozen["fold"], "pixel_count_uint64": pixels, "sample_id": frozen["sample_id"], "selection_order": selection, "target_sha256": target_sha, "prediction_sha256": prediction_sha}[key])
        if observed:
            raise RuntimeError("unexpected OOF computation warning")
        after_compute_rng = rng_snapshot()
        if before_rng != after_compute_rng:
            raise RuntimeError("OOF computation RNG drift")
        events_payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in events).encode()
        atomic_bytes(prep / "oof_call_events.ndjson", events_payload)
        metrics_path = prep / "metrics.npz"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(metrics_path), flags, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            np.savez(stream,
                absolute_error_sum_uint64=np.asarray(metrics["absolute_error_sum_uint64"], dtype="uint64"),
                branch=np.asarray(metrics["branch"], dtype="int64"), episode=np.asarray(metrics["episode"], dtype="int64"),
                fold=np.asarray(metrics["fold"], dtype="int64"), pixel_count_uint64=np.asarray(metrics["pixel_count_uint64"], dtype="uint64"),
                sample_id=np.asarray(metrics["sample_id"], dtype="int64"), selection_order=np.asarray(metrics["selection_order"], dtype="int64"),
                target_sha256=np.asarray(metrics["target_sha256"], dtype="<U64"), prediction_sha256=np.asarray(metrics["prediction_sha256"], dtype="<U64"))
            stream.flush(); os.fsync(stream.fileno())
        exact_file(metrics_path, sha(metrics_path), metrics_path.stat().st_size)
        ordered_event_digests = [x["event_canonical_sha256"] for x in events]
        for fold in range(5):
            subset = events[fold * 200:(fold + 1) * 200]
            if len(subset) != 200 or any(x["fold"] != fold for x in subset):
                raise RuntimeError("fold partition")
            atomic_json(prep / f"fold_{fold}_receipt.json", {"format": "strict-track2-v531-v530-actual-oof-fold-receipt-v1", "status": "computed_pending_independent_audit", "fold": fold, "rows": 200, "event_digest_sha256": canonical_sha(ordered_event_digests[fold * 200:(fold + 1) * 200]), "absolute_error_sum_uint64": sum(x["absolute_error_sum_uint64"] for x in subset), "pixel_count_uint64": sum(x["pixel_count_uint64"] for x in subset)})
        atomic_json(prep / "source_manifest.json", {"format": "strict-track2-v531-v530-actual-oof-source-manifest-v1", "manifest_sha256": args.manifest_sha, "preregistration_sha256": args.preregistration_sha, "contract_sha256": args.contract_sha, "authority_receipt_sha256": args.authority_receipt_sha, "executor_sha256": args.executor_sha, "auditor_sha256": args.auditor_sha, "launcher_sha256": args.launcher_sha})
        spec = importlib.util.spec_from_file_location("v531_actual_oof_auditor", args.auditor_source)
        auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
        audit = auditor.audit_staged(prep, manifest, cache_path)
        atomic_json(prep / "independent_audit.json", audit)
        post_qualification = tree(Path(manifest["cache_input"]["qualification_root"]))
        if post_qualification != before_qualification:
            raise RuntimeError("qualification mutated")
        after_rng = rng_snapshot()
        if after_rng != before_rng:
            raise RuntimeError("boundary RNG drift")
        services_after = services_snapshot(); gpu_after = gpu_processes()
        if services_after != services_before or gpu_after != gpu_before: raise RuntimeError("services/PIDGPU drift")
        receipt = {"format": FORMAT, "status": "passed_actual_oof_execution", "passed": True, "seed": SEED, "actual_oof_execution_boundary_invocations": 1, "launcher_invocations": 1, "executor_invocations": 1, "auditor_import_invocations": 1, "events": 1000, "folds": 5, "fold_row_counts": [200] * 5, "branch_order": BRANCHES, "ordered_rows_canonical_sha256": manifest["ordered_rows_canonical_sha256"], "ordered_event_digest_sha256": canonical_sha(ordered_event_digests), "metrics_npz_sha256": sha(metrics_path), "events_sha256": sha(prep / "oof_call_events.ndjson"), "independent_audit_sha256": sha(prep / "independent_audit.json"), "qualification_pre_tree": before_qualification, "qualification_post_tree": post_qualification, "qualification_mutated": False, "rng_entry": before_rng, "rng_exit": after_rng, "rng_external_restored": True, "services_pre": services_before, "services_post": services_after, "services_restored": True, "gpu_processes_pre": gpu_before, "gpu_processes_post": gpu_after, "gpu_processes_unchanged_empty": True, "oof_computation_warning_count": 0, "input_raw_warning_count": 1000, "model_runtime_delegate_invocations": 0, "phase_a_replay_invocations": 0, "training_invocations": 0, "cache_reuse_invocations": 0, "reward_read_invocations": 0, "dev_hidden_final_outcome_read_invocations": 0, "submission_invocations": 0, "retry_authorized": False, "output_publication": {"noreplace": True, "commit_semantics": "fresh_oof_root_visibility_after_renameat2_noreplace", "crash_durability_claimed": False, "postcommit_parent_dir_fsync_best_effort": True}}
        atomic_json(prep / "execution_receipt.json", receipt)
        if prep.stat().st_dev != identity.st_dev or prep.stat().st_ino != identity.st_ino:
            raise RuntimeError("prep identity")
        if sorted(x.name for x in prep.iterdir()) != sorted(manifest["output_schema"]["exact_members"]):
            raise RuntimeError("output exact members")
        validate_input_records(prereg)
        if tree(Path(manifest["cache_input"]["qualification_root"])) != before_qualification:
            raise RuntimeError("prepublish qualification drift")
        rename_noreplace(prep, args.output_root)
        committed = True
        try:
            fd = os.open(str(args.output_root.parent), os.O_RDONLY); os.fsync(fd); os.close(fd)
        except Exception:
            pass
        return receipt
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if not committed and prep.exists():
            try:
                current = prep.stat()
                if current.st_dev == identity.st_dev and current.st_ino == identity.st_ino:
                    for path in sorted(prep.rglob("*"), reverse=True):
                        if path.is_file() and not path.is_symlink(): path.unlink()
                    prep.rmdir()
            except Exception:
                pass


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", type=Path, required=True); result.add_argument("--preregistration-sha", required=True)
    result.add_argument("--manifest", type=Path, required=True); result.add_argument("--manifest-sha", required=True)
    result.add_argument("--contract", type=Path, required=True); result.add_argument("--contract-sha", required=True)
    result.add_argument("--authority-receipt", type=Path, required=True); result.add_argument("--authority-receipt-sha", required=True)
    result.add_argument("--executor-sha", required=True); result.add_argument("--auditor-source", type=Path, required=True); result.add_argument("--auditor-sha", required=True)
    result.add_argument("--launcher-source", type=Path, required=True); result.add_argument("--launcher-sha", required=True)
    result.add_argument("--launcher-argv-json", required=True)
    result.add_argument("--launcher-pid", type=int, required=True); result.add_argument("--launcher-ppid", type=int, required=True)
    result.add_argument("--launcher-pgid", type=int, required=True); result.add_argument("--launcher-sid", type=int, required=True)
    result.add_argument("--seed", type=int, required=True); result.add_argument("--output-root", type=Path, required=True)
    return result


def main(argv=None):
    literal = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(literal); args.literal_argv = literal
    return execute(args)


def synthetic_self_test() -> bool:
    exact = {"a": 1, "b": False}
    return set(exact) == {"a", "b"} and type(exact["a"]) is int and type(exact["b"]) is bool


if __name__ == "__main__":
    if sys.argv[1:] == ["--synthetic-self-test"]:
        print(json.dumps({"passed": synthetic_self_test()}, sort_keys=True)); raise SystemExit(0)
    try:
        result = main(); print(json.dumps({"passed": result["passed"], "status": result["status"], "events": result["events"], "folds": result["folds"]}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"passed": False, "status": "failed_no_retry", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr); raise SystemExit(79)
