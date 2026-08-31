#!/usr/bin/env python3
"""Materialize the fresh v506 compact authority after the v505 split terminal."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import stat
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
SCRIPTS = ROOT / "pipeline/scripts"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SELF_PATH = SCRIPTS / "materialize_v506_v505_v503_compact_reconciliation_execution_authority.py"
BASE_PATH = SCRIPTS / "materialize_v503_v502_compact_reconciliation_execution_authority.py"
BASE_SHA = "bfdf33ab525d3b9e98ea116604f309354160cf98021478c94c177caa5060d879"
BASE_BYTES = 42234
CONTRACT_PATH = SCRIPTS / "v506_v505_v503_compact_reconciliation_execution_authority_contract.json"
WRAPPER_PATH = SCRIPTS / "launch_v506_v505_v503_compact_single_reconciler.py"
AUTHORITY_ROOT = J / "v506_v505_v503_compact_reconciliation_execution_authority_seed1647_20260825"
WRAPPER_ATTEMPT_ROOT = J / "v506_v505_v503_compact_single_reconciler_attempt_seed1647_20260825"
SPLIT_FORENSIC_PATH = SCRIPTS / "v506_v505_v503_authority_transport_split_state_forensic.json"
OLD_AUTHORITY_ROOT = J / "v503_v502_compact_reconciliation_execution_authority_seed1645_20260825"
OLD_EVIDENCE_ROOT = J / "v505_v503_compact_reconciliation_execution_authority_materialization_evidence_seed1646_20260825"

CONTRACT_FORMAT = "strict-track2-v506-v505-v503-compact-reconciliation-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v506-v505-v503-compact-reconciliation-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_v506_compact_single_reconciler_attempt"

CONTRACT_SOURCE_ORDER = [
    "authority_materializer", "base_authority_materializer", "direct_reconciler_wrapper",
    "reconciler_r2", "phase_a_design_contract", "repair_formal_receipt",
    "postregistration_static_receipt",
]
AUTHORITY_SOURCE_ORDER = ["authority_design_contract", *CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES = {
    "authority_design_contract": "authority_design_contract",
    "authority_materializer": "authority_materializer_source",
    "base_authority_materializer": "base_authority_materializer_source",
    "direct_reconciler_wrapper": "direct_reconciler_wrapper_source",
    "reconciler_r2": "reconciler_r2_source",
    "phase_a_design_contract": "phase_a_design_contract_source",
    "repair_formal_receipt": "repair_formal_source",
    "postregistration_static_receipt": "postregistration_static_source",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_role_order",
    "source_aliases", "source_closure_sha256", "authority_materializer_source",
    "base_authority_materializer_source", "repair_formal_record", "postregistration_static_record",
    "phase_a_design_contract_record", "f813_registration_tree", "v505_split_state_forensic",
    "v503_materialized_authority_receipt", "v503_materialized_authority_registration_tree",
    "v505_materialization_evidence_tree", "v505_materialization_process_receipt",
    "v505_materialization_stdout", "v505_split_state_transition", "historical_absences",
    "current_absences_after_authority", "authority_receipt_contract", "authorization",
    "runtime_observation", "execution_boundary",
}
CHECK_KEYS = sorted({
    "authority_contract_current", "authority_materializer_current", "base_materializer_current",
    "direct_wrapper_current", "r2_current", "phase_contract_current", "repair_formal_current",
    "static_receipt_current", "source_closure_current", "source_order_exact", "source_aliases_exact",
    "f813_exact2", "v505_split_forensic_exact", "v503_authority_exact1",
    "v503_authority_schema_exact", "v505_evidence_exact6", "v505_process_failed_no_retry",
    "v505_stdout_exact2", "v505_materializer_native_empty", "v505_split_partition_exact",
    "v505_split_transition_exact", "historical_absences", "current_absences",
    "input_snapshots_equal", "no_live_process", "gpu_empty", "authorization_boundary",
    "execution_boundary", "no_pending_values",
})
AUTH_TOP_KEYS = {
    "format", "status", "passed", "source_closure", "source_role_order", "source_aliases",
    "source_closure_sha256", "authority_design_contract", "authority_materializer_source",
    "base_authority_materializer_source", "direct_reconciler_wrapper_source", "reconciler_r2_source",
    "phase_a_design_contract_source", "repair_formal_source", "postregistration_static_source",
    "repair_formal_record", "postregistration_static_record", "phase_a_design_contract_record",
    "f813_registration_tree", "v505_split_state_forensic", "v503_materialized_authority_receipt",
    "v503_materialized_authority_registration_tree", "v505_materialization_evidence_tree",
    "v505_materialization_process_receipt", "v505_materialization_stdout",
    "v505_split_state_transition", "wrapper_attempt_root", "transparent_static_receipt_path",
    "historical_absences", "required_absences", "checks", "check_keys", "check_key_set_sha256",
    "checks_sha256", "input_pre_snapshot", "input_post_snapshot", "input_snapshots_exactly_equal",
    "authorization", "runtime_observation", "execution_boundary",
}

AUTHORIZATION = {
    "direct_single_reconciler_authorized": True,
    "direct_single_reconciler_invocations_authorized": 1,
    "direct_single_reconciler_invocations_consumed": 0,
    "direct_r2_authorized": False,
    "nested_r2_invocations_authorized": 1,
    "nested_r2_only_via_single_reconciler": True,
    "retry_authorized": False,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "folds_authorized": 0,
    "policy_updates": 0,
    "rl_authorized": False,
    "submission_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME = {
    "execution_authority_materialized": True,
    "direct_single_reconciler_executed": False,
    "reconciler_r2_executed": False,
    "transparent_receipt_created": False,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True,
    "direct_single_reconciler_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_invocations": 0,
    "training_invocations": 0,
    "reward_reads": 0,
    "dev_hidden_final_outcome_reads": 0,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def regular(path: Path, digest: str | None = None, logical_bytes: int | None = None) -> dict:
    metadata = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("not regular: " + str(path))
    value = {"path": str(path), "sha256": sha(path), "logical_bytes": metadata.st_size}
    if digest is not None and value["sha256"] != digest:
        raise RuntimeError("sha: " + str(path))
    if logical_bytes is not None and value["logical_bytes"] != logical_bytes:
        raise RuntimeError("bytes: " + str(path))
    return value


def load_base():
    regular(BASE_PATH, BASE_SHA, BASE_BYTES)
    spec = importlib.util.spec_from_file_location("v503_frozen_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("base module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_base(base) -> None:
    base.CONTRACT_PATH = CONTRACT_PATH
    base.MATERIALIZER_PATH = SELF_PATH
    base.WRAPPER_PATH = WRAPPER_PATH
    base.AUTHORITY_ROOT = AUTHORITY_ROOT
    base.WRAPPER_ATTEMPT_ROOT = WRAPPER_ATTEMPT_ROOT
    base.CONTRACT_FORMAT = CONTRACT_FORMAT
    base.CONTRACT_STATUS = CONTRACT_STATUS
    base.OUTPUT_FORMAT = OUTPUT_FORMAT
    base.OUTPUT_STATUS = OUTPUT_STATUS
    base.AUTHORIZATION = AUTHORIZATION
    base.RUNTIME = RUNTIME
    base.EXECUTION_BOUNDARY = EXECUTION_BOUNDARY
    base.CONTRACT_SOURCE_ORDER = CONTRACT_SOURCE_ORDER
    base.AUTHORITY_SOURCE_ORDER = AUTHORITY_SOURCE_ORDER
    base.SOURCE_ALIASES = SOURCE_ALIASES
    base.CONTRACT_TOP_KEYS = CONTRACT_TOP_KEYS
    base.AUTH_TOP_KEYS = AUTH_TOP_KEYS
    base.CHECK_KEYS = CHECK_KEYS


def record(base, value) -> dict:
    return base.record(value)


def build_context(base, args):
    if args.contract != CONTRACT_PATH or args.materializer_source != SELF_PATH or args.authority_root != AUTHORITY_ROOT:
        raise RuntimeError("canonical CLI")
    contract_record = regular(args.contract, args.contract_sha, args.contract_bytes)
    materializer_record = regular(args.materializer_source, args.materializer_sha, args.materializer_bytes)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1647
            or base.pending(contract)):
        raise RuntimeError("contract schema")
    if contract["authority_materializer_source"] != materializer_record:
        raise RuntimeError("materializer binding")
    sources = {name: record(base, row) for name, row in contract["source_closure"].items()}
    if (set(sources) != set(CONTRACT_SOURCE_ORDER) or contract["source_role_order"] != CONTRACT_SOURCE_ORDER
            or contract["source_aliases"] != {key: SOURCE_ALIASES[key] for key in CONTRACT_SOURCE_ORDER}
            or contract["source_closure_sha256"] != base.csha(sources)
            or sources["authority_materializer"] != materializer_record):
        raise RuntimeError("source closure")
    fixed = {
        "base_authority_materializer": (BASE_PATH, BASE_SHA, BASE_BYTES),
        "reconciler_r2": (base.R2_PATH, "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377", 51845),
        "phase_a_design_contract": (base.PHASE_PATH, "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64", 43960),
        "repair_formal_receipt": (base.REPAIR_PATH, "b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b", 36181),
        "postregistration_static_receipt": (base.STATIC_PATH, "441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb", 49008),
    }
    for role, (path, digest, size) in fixed.items():
        if sources[role] != {"path": str(path), "sha256": digest, "logical_bytes": size}:
            raise RuntimeError("fixed source " + role)
    if Path(sources["direct_reconciler_wrapper"]["path"]) != WRAPPER_PATH:
        raise RuntimeError("wrapper path")
    if (contract["base_authority_materializer_source"] != sources["base_authority_materializer"]
            or contract["repair_formal_record"] != sources["repair_formal_receipt"]
            or contract["postregistration_static_record"] != sources["postregistration_static_receipt"]
            or contract["phase_a_design_contract_record"] != sources["phase_a_design_contract"]):
        raise RuntimeError("source aliases")
    f813 = base.verify_tree(contract["f813_registration_tree"])
    if (f813["root"] != str(base.F813_ROOT) or f813["file_count"] != 2
            or f813["logical_file_bytes"] != 27771
            or f813["sha256sum_lines_digest_sha256"] != "717d6ae12fbacbaafa147d03503140e5f807c028acc97a7294c84da2d79fe3fe"
            or f813["canonical_json_triples_digest_sha256"] != "a0602666a7e59d60e464b3bc76b31b2fbf501e3d37bbd0b18b2db8c1233971ec"):
        raise RuntimeError("F813 exact2")
    historical_keys = {
        "execution_authority_root", "execution_authority_prep", "wrapper_attempt_root",
        "wrapper_attempt_prep", "transparent_receipt", "transparent_receipt_tmp",
        "qualification_root", "v503_wrapper_attempt_root", "v503_wrapper_attempt_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = base.decode_absences(contract["historical_absences"], historical_keys)
    current = base.decode_absences(contract["current_absences_after_authority"], current_keys)
    old_wrapper_root = J / "v503_v502_compact_single_reconciler_attempt_seed1645_20260825"
    expected = {
        "execution_authority_root": AUTHORITY_ROOT,
        "execution_authority_prep": AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep"),
        "wrapper_attempt_root": WRAPPER_ATTEMPT_ROOT,
        "wrapper_attempt_prep": WRAPPER_ATTEMPT_ROOT.with_name(WRAPPER_ATTEMPT_ROOT.name + ".attempt-prep"),
        "transparent_receipt": base.TRANSPARENT_PATH,
        "transparent_receipt_tmp": base.TRANSPARENT_PATH.with_name(base.TRANSPARENT_PATH.name + ".tmp"),
        "qualification_root": base.QUALIFICATION_ROOT,
        "v503_wrapper_attempt_root": old_wrapper_root,
        "v503_wrapper_attempt_prep": old_wrapper_root.with_name(old_wrapper_root.name + ".attempt-prep"),
    }
    if historical != expected or current != {key: value for key, value in expected.items() if key in current_keys}:
        raise RuntimeError("absence paths")
    if any(os.path.lexists(path) for path in historical.values()):
        raise RuntimeError("absence prestate")
    if base.live_processes({str(WRAPPER_PATH), str(base.R2_PATH), str(WRAPPER_ATTEMPT_ROOT)}) or not base.gpu_empty():
        raise RuntimeError("process or GPU prestate")
    authority_sources = {"authority_design_contract": contract_record, **sources}
    files = {
        "contract": str(CONTRACT_PATH), "materializer": str(SELF_PATH), "base_materializer": str(BASE_PATH),
        "wrapper": str(WRAPPER_PATH), "r2": str(base.R2_PATH), "phase": str(base.PHASE_PATH),
        "repair": str(base.REPAIR_PATH), "static": str(base.STATIC_PATH),
    }
    trees = {"f813": str(base.F813_ROOT)}
    return contract, contract_record, materializer_record, authority_sources, historical, current, {
        "files": files, "trees": trees, "f813": f813,
    }


def verify_split(base, contract: dict) -> dict:
    base_record = record(base, contract["base_authority_materializer_source"])
    if base_record != {"path": str(BASE_PATH), "sha256": BASE_SHA, "logical_bytes": BASE_BYTES}:
        raise RuntimeError("base materializer source")
    if contract["source_closure"].get("base_authority_materializer") != base_record:
        raise RuntimeError("base source closure")
    forensic_record = record(base, contract["v505_split_state_forensic"])
    forensic = json.loads(Path(forensic_record["path"]).read_text())
    old_receipt = record(base, contract["v503_materialized_authority_receipt"])
    old_tree = base.verify_tree(contract["v503_materialized_authority_registration_tree"])
    evidence_tree = base.verify_tree(contract["v505_materialization_evidence_tree"])
    process_record = record(base, contract["v505_materialization_process_receipt"])
    stdout_record = record(base, contract["v505_materialization_stdout"])
    old_value = json.loads(Path(old_receipt["path"]).read_text())
    process = json.loads(Path(process_record["path"]).read_text())
    stdout_lines = [json.loads(line) for line in Path(stdout_record["path"]).read_text().splitlines()]
    transition = {
        "authority_materialized": True,
        "authority_receipt_passed": True,
        "transport_helper_terminal_passed": False,
        "transport_helper_failure_stage": "post_authority_snapshot_revalidated_pre_authority_absence",
        "authority_materializer_invocations": 1,
        "direct_single_reconciler_invocations": 0,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "fresh_authority_required": True,
    }
    if (
        forensic_record["path"] != str(SPLIT_FORENSIC_PATH)
        or forensic.get("passed") is not True
        or forensic_record["sha256"] != "f1a383247039513c8657afe5da8395024200fec66869fb417d459f988821a003"
        or forensic_record["logical_bytes"] != 7557
        or forensic.get("status") != "authority_materialized_exact_once_transport_failed_no_wrapper_execution"
        or forensic.get("state_transition", {}).get("sole_changed_runtime_root_predicate") is not True
        or forensic.get("state_transition", {}).get("other_prior_forensic_predicates_still_true") is not True
        or old_receipt != {"path": str(OLD_AUTHORITY_ROOT / "authority_receipt.json"), "sha256": "394c65ae0ca0979b0ee3c9c774e3b69d68b7ff5e5b1e95e17f02da74c06b223b", "logical_bytes": 46959}
        or old_tree["root"] != str(OLD_AUTHORITY_ROOT) or old_tree["file_count"] != 1
        or old_tree["sha256sum_lines_digest_sha256"] != "9ea3295006d6903176470df5f39933085721ab86ae779a29f7361db5b00e7489"
        or old_tree["canonical_json_triples_digest_sha256"] != "d5ca54ebee9f041c7509f8eb929ffb00e5c7ef9f804d4cb74d2963b448739f6a"
        or len(old_value) != 46 or old_value.get("passed") is not True
        or len(old_value.get("checks", {})) != 33 or not all(old_value.get("checks", {}).values())
        or len(old_value.get("source_closure", {})) != 7
        or old_value.get("input_snapshots_exactly_equal") is not True
        or evidence_tree["root"] != str(OLD_EVIDENCE_ROOT) or evidence_tree["file_count"] != 6
        or evidence_tree["logical_file_bytes"] != 78049
        or evidence_tree["sha256sum_lines_digest_sha256"] != "8fc76a710e3b69710ff5e18c10e379ea7eb6c85f8050d2dd03b2aa3a85e699a3"
        or evidence_tree["canonical_json_triples_digest_sha256"] != "fa123c6edab77c8ee3b307dadb15e4bfe0056aa501dc8c2edbb58dac609ed450"
        or process_record != {"path": str(OLD_EVIDENCE_ROOT / "process_receipt.json"), "sha256": "ff2b4b11c47cdd255b61b6941692f2793bf8e6125a0c93b70a8cea2f981fac59", "logical_bytes": 2982}
        or stdout_record != {"path": str(OLD_EVIDENCE_ROOT / "materializer_stdout.log"), "sha256": "1f15250dbb9153126cba342f9166fa2472e00a48d7e6a903f679673741338eae", "logical_bytes": 1132}
        or process.get("status") != "failed_no_retry" or process.get("passed") is not False
        or process.get("transport_helper_invocations") != 1 or process.get("authority_materializer_invocations") != 1
        or process.get("direct_single_reconciler_invocations") != 0 or process.get("reconciler_r2_invocations") != 0
        or process.get("retry_authorized") is not False
        or process.get("cleanup", {}).get("reaped") is not True or process.get("cleanup", {}).get("group_empty") is not True
        or len(stdout_lines) != 2
        or stdout_lines[0].get("line_origin") != "transport_helper" or stdout_lines[0].get("materializer_native_empty") is not True
        or stdout_lines[0].get("authority_receipt") != old_receipt
        or stdout_lines[1] != {"authority_materializer_invocations": 1, "committed_success": True,
                               "direct_single_reconciler_invocations": 0, "line_origin": "transport_helper",
                               "passed": True, "reconciler_r2_invocations": 0}
        or contract["v505_split_state_transition"] != transition
    ):
        raise RuntimeError("v505 split closure")
    return {
        "base": base_record, "forensic": forensic_record, "old_receipt": old_receipt,
        "old_tree": old_tree, "evidence_tree": evidence_tree, "process": process_record,
        "stdout": stdout_record, "transition": transition,
    }


def make_receipt(base, contract, contract_record, materializer_record, sources, historical, current, anchors, split, pre, post):
    return {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "source_closure": sources, "source_role_order": base.AUTHORITY_SOURCE_ORDER,
        "source_aliases": base.SOURCE_ALIASES, "source_closure_sha256": base.csha(sources),
        "authority_design_contract": contract_record, "authority_materializer_source": materializer_record,
        "base_authority_materializer_source": split["base"],
        "direct_reconciler_wrapper_source": sources["direct_reconciler_wrapper"],
        "reconciler_r2_source": sources["reconciler_r2"],
        "phase_a_design_contract_source": sources["phase_a_design_contract"],
        "repair_formal_source": sources["repair_formal_receipt"],
        "postregistration_static_source": sources["postregistration_static_receipt"],
        "repair_formal_record": contract["repair_formal_record"],
        "postregistration_static_record": contract["postregistration_static_record"],
        "phase_a_design_contract_record": contract["phase_a_design_contract_record"],
        "f813_registration_tree": anchors["f813"],
        "v505_split_state_forensic": split["forensic"],
        "v503_materialized_authority_receipt": split["old_receipt"],
        "v503_materialized_authority_registration_tree": split["old_tree"],
        "v505_materialization_evidence_tree": split["evidence_tree"],
        "v505_materialization_process_receipt": split["process"],
        "v505_materialization_stdout": split["stdout"], "v505_split_state_transition": split["transition"],
        "wrapper_attempt_root": str(WRAPPER_ATTEMPT_ROOT),
        "transparent_static_receipt_path": str(base.TRANSPARENT_PATH),
        "historical_absences": {k: {"path": str(v), "absent": True} for k, v in sorted(historical.items())},
        "required_absences": {k: {"path": str(v), "absent": True} for k, v in sorted(current.items())},
        "checks": {k: True for k in base.CHECK_KEYS}, "check_keys": base.CHECK_KEYS,
        "check_key_set_sha256": base.csha(base.CHECK_KEYS),
        "checks_sha256": base.csha({k: True for k in base.CHECK_KEYS}),
        "input_pre_snapshot": pre, "input_post_snapshot": post, "input_snapshots_exactly_equal": True,
        "authorization": AUTHORIZATION, "runtime_observation": RUNTIME, "execution_boundary": EXECUTION_BOUNDARY,
    }


def synthetic() -> int:
    checks = {
        "seed": 1647 == 1647,
        "fresh_paths": all("v506_" in path.name for path in (CONTRACT_PATH, SELF_PATH, WRAPPER_PATH, AUTHORITY_ROOT, WRAPPER_ATTEMPT_ROOT)),
        "base_frozen": len(BASE_SHA) == 64 and BASE_BYTES == 42234,
        "authorization": AUTHORIZATION["direct_single_reconciler_authorized"] is True
                         and AUTHORIZATION["direct_r2_authorized"] is False
                         and AUTHORIZATION["retry_authorized"] is False,
        "contract_top_count": len(CONTRACT_TOP_KEYS) == 27,
        "authority_top_count": len(AUTH_TOP_KEYS) == 40,
        "check_count": len(CHECK_KEYS) == 29,
        "source_counts": len(CONTRACT_SOURCE_ORDER) == 7 and len(AUTHORITY_SOURCE_ORDER) == 8,
    }
    print(json.dumps({"passed": all(checks.values()), "checks": checks, "checks_sha256": csha(checks)}, sort_keys=True))
    return 0 if all(checks.values()) else 1


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--contract-bytes", type=int, required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--materializer-bytes", type=int, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--read-only-preflight", action="store_true")
    args = parser.parse_args()
    base = load_base()
    configure_base(base)
    contract, contract_record, materializer_record, sources, historical, current, anchors = build_context(base, args)
    split = verify_split(base, contract)
    anchors["files"].update({
        "base_materializer": str(BASE_PATH), "split_forensic": split["forensic"]["path"],
        "old_authority_receipt": split["old_receipt"]["path"], "v505_process": split["process"]["path"],
        "v505_stdout": split["stdout"]["path"],
    })
    anchors["trees"].update({"old_authority": str(OLD_AUTHORITY_ROOT), "v505_evidence": str(OLD_EVIDENCE_ROOT)})
    pre = base.snapshot(anchors["files"], anchors["trees"], current)
    post = base.snapshot(anchors["files"], anchors["trees"], current)
    if pre != post:
        raise RuntimeError("input drift")
    receipt = make_receipt(base, contract, contract_record, materializer_record, sources, historical, current,
                           anchors, split, pre, post)
    schema = contract["authority_receipt_contract"]
    if (set(receipt) != base.AUTH_TOP_KEYS or schema.get("format") != OUTPUT_FORMAT
            or schema.get("status") != OUTPUT_STATUS or schema.get("top_keys") != sorted(base.AUTH_TOP_KEYS)
            or schema.get("check_keys") != base.CHECK_KEYS
            or schema.get("check_key_set_sha256") != base.csha(base.CHECK_KEYS)
            or schema.get("checks_sha256") != base.csha({k: True for k in base.CHECK_KEYS})
            or schema.get("authorization_exact") != AUTHORIZATION
            or schema.get("runtime_observation_exact") != RUNTIME
            or contract["authorization"] != AUTHORIZATION or contract["runtime_observation"] != RUNTIME
            or contract["execution_boundary"] != EXECUTION_BOUNDARY):
        raise RuntimeError("fresh receipt schema")
    if args.read_only_preflight:
        print(json.dumps({"status": "passed_read_only_preflight_no_materialization",
                          "contract_top_count": len(contract), "authority_top_count": len(receipt),
                          "check_count": len(base.CHECK_KEYS), "contract_source_count": len(contract["source_closure"]),
                          "authority_source_count": len(sources), "input_snapshots_equal": pre == post}, sort_keys=True))
        return 0
    stable_absences = {k: v for k, v in current.items() if k != "execution_authority_prep"}
    stable_pre = base.snapshot(anchors["files"], anchors["trees"], stable_absences)
    base.publish_exact1(AUTHORITY_ROOT, receipt, anchors["files"], anchors["trees"], stable_absences, stable_pre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
