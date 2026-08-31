#!/usr/bin/env python3
"""One-shot, fail-closed launcher for the v486 read-only reconciliation.

This launcher never imports v169, torch, a model, a reward implementation, or
training code.  It writes attempt evidence outside the frozen f813
registration, then invokes the frozen c71 reconciler exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if os.name == "posix":
    import fcntl
else:  # pragma: no cover - production execution is Linux; Windows is compile/fixture only.
    fcntl = None

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
FORMAL_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json"
FORMAL_SHA = "f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8"
FORMAL_BYTES = 21296
AUTHORITY_ROOT = J / "v486_v485_phase_a_static_reconciliation_authority_seed1628_20260824"
AUTHORITY_PATH = AUTHORITY_ROOT / "authority_receipt.json"
AUTHORITY_DESIGN_PATH = ROOT / "pipeline/scripts/v486_v485_phase_a_static_reconciliation_postregistration_authority_contract.json"
AUTHORITY_DESIGN_FORMAT = "strict-track2-v486-v485-phase-a-static-reconciliation-postregistration-authority-design-contract-v1"
AUTHORITY_DESIGN_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
AUTHORITY_MATERIALIZER_PATH = ROOT / "pipeline/scripts/materialize_v486_v485_phase_a_static_reconciliation_postregistration_authority.py"
AUTHORITY_MATERIALIZER_SHA = "33079cd1827214c851b771f303dd5b575a12699116de7f927a18c5ea25ba7220"
AUTHORITY_MATERIALIZER_BYTES = 20222
ATTEMPT_ROOT = J / "v486_v485_phase_a_static_reconciliation_attempt_seed1628_20260824"
ATTEMPT_PREP = ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + ".attempt-prep")
TRANSPARENT_OUTPUT = FORMAL_PATH.parent / "transparent_static_audit.json"
RECON_CONTRACT_PATH = ROOT / "pipeline/scripts/v486_v485_phase_a_static_false_positive_reconciliation_authority_contract.json"
RECON_CONTRACT_SHA = "701236b3e0bfde3eb6e70758891df37370b2b99b151eb31850830b67856049fa"
RECON_CONTRACT_BYTES = 14867
PHASE_A_CONTRACT_PATH = ROOT / "pipeline/scripts/v485_v482_v169_cache_determinism_scope_repair_contract.json"
PHASE_A_CONTRACT_SHA = "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
RECONCILER_PATH = ROOT / "pipeline/scripts/reconcile_v486_v485_static_false_positive.py"
RECONCILER_SHA = "c71ba00b0c92efda03f9df149263d0c476de86ea7338ba3719a72f3305959984"
RECONCILER_BYTES = 36229
MATERIALIZER_PATH = ROOT / "pipeline/scripts/materialize_v486_v485_phase_a_static_false_positive_reconciliation_preregistration.py"
MATERIALIZER_SHA = "942ed1d83dd994698264bc21a2a627b05f2c42df6e140293d8617fead50376f1"
MATERIALIZER_BYTES = 19128
RLPY_LEXICAL = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
RLPY_RESOLVED = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
RLPY_SHA = "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788"
RLPY_BYTES = 25555040
OLD_FORMAL_PATH = J / "v485_v169_cache_qualification_prereg_seed1627_20260824/preregistration.json"
OLD_FORMAL_SHA = "b73fad8071d8df0b1bed3dc53212bf8a86b2350911620feac81d96a1505492e0"
OLD_STATIC_SOURCE = ROOT / "pipeline/scripts/audit_v485_v169_cache_qualification_static.py"
OLD_STATIC_SHA = "d4cadd94141378f0cf89892847894abefc91357304f44d470698221a6613fc5f"
OLD_STATIC_RECEIPT = OLD_FORMAL_PATH.parent / "static_audit.json"
OLD_STATIC_RECEIPT_SHA = "4a287cd3b92ff281a785f454f8844e9695a58fd4b1de277e054ab4c22df48afc"
OLD_STATIC_LOG = Path("/dev/shm/v485_static_b73.log")
OLD_STATIC_LOG_SHA = "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b"
PERSISTENT_OLD_STATIC_LOG = FORMAL_PATH.parent / "immutable_evidence/v485_static_b73.log"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")
AUTH_FORMAT = "strict-track2-v486-v485-phase-a-static-reconciliation-postregistration-authority-v1"
AUTH_STATUS = "authorized_exact_one_readonly_static_reconciliation_attempt"
INTENT_FORMAT = "strict-track2-v486-v485-static-reconciliation-attempt-intent-v1"
TERMINAL_FORMAT = "strict-track2-v486-v485-static-reconciliation-attempt-terminal-v1"
AUTHORIZATION = {
    "readonly_static_reconciliation_authorized": True,
    "attempts_authorized": 1,
    "attempts_consumed": 0,
    "retry_authorized": False,
    "phase_a_cache_qualification_authorized": False,
    "cache_reuse_authorized": False,
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
RUNTIME_PRE = {
    "reconciliation_executed": False,
    "phase_a_executed": False,
    "v169_imported_or_run": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
FALSE_AUTHORITIES = {
    "retry_authorized": False,
    "phase_a_cache_qualification_authorized": False,
    "cache_reuse_authorized": False,
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
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CHECK_KEYS = (
    "A_acyclic_receipt_payload_manifest",
    "A_nested_fsync_and_whole_promote",
    "CPU_synthetic_fixtures",
    "all_sources_exact",
    "auditor_completion_and_cleanup_crosslinks",
    "auditor_exact_schema_and_AB_bytes",
    "auditor_independent_rebuild_full_closure",
    "contract_design_only",
    "contract_exact",
    "determinism_scope_exact",
    "driver_actual_rehash_exact7_roles",
    "driver_completion_real_dispatch_fixture",
    "driver_exact_A_then_B_then_audits",
    "driver_failure_intent_stage_completion",
    "driver_owned_full_lifetime_logs_and_completion",
    "driver_run_signature_and_reachable_audits",
    "driver_staged_terminal_trees",
    "exact_raw_dtypes_no_masking",
    "execution_interpreter_actual",
    "final_auditor_rehashes_full_closure",
    "formal_contract_sections_exact",
    "formal_exact",
    "formal_nonauthorizing",
    "formal_source_records_digest",
    "formal_sources_exact",
    "fresh_tuple_and_identity_after_failure_fixture",
    "helper_exact_warning",
    "identity_before_v169",
    "intent_nonce_roles_reverse_binding",
    "intent_rename_failure_boundary",
    "launcher_argument_driven_self_bound_zero_state",
    "launcher_authority_and_driver_exact",
    "launcher_runtime_cleanup_resources",
    "materializer_whole_dir_and_sections",
    "no_training_reward_outcome",
    "output_root_only_formal_authority",
    "parent_runtime_path_SHA_before_import_and_independent_audit",
    "qualification_root_absent",
    "scope_bootstrap_verified_bytes_before_import",
    "scope_has_no_parent_runtime_import",
    "static_failed_cache_state",
    "static_parent_files_absences",
    "static_parent_tree_rehash",
    "static_self_bound",
    "static_small_smoke_rehash",
    "terminal_unique_quiet_commit",
    "worker_progress_events_fsynced",
)
CHECK_KEYSET_SHA = "57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b"
ALL_TRUE_CHECKS_SHA = "c1697befd4fdc6a155e709b54ef98f3679da78a6978bf2630d1c82462477176e"
SOURCE_RECORDS_SHA = "9adc5bdbfaa0022b745c44abac2ac2bd02f801f751ebce35d0116c0a8e8e53e0"
AUTHORITY_CHECK_KEYS = (
    "all_required_outputs_absent",
    "exact7_current",
    "first_failure_disclosed_non_native_reconstructed",
    "formal_exact_nonauthorizing_phase_a",
    "input_pre_post_snapshots_equal",
    "no_reconciliation_process",
    "old_b73_REG_and_44_3_exact",
    "persistent_and_volatile_log_exact",
    "reconciler_and_wrapper_source_exact",
    "reconciliation_REG_exact2",
    "successful_native_logs_explicitly_unavailable",
    "successful_transport_stdout_recomputed",
)
AUTHORITY_CHECK_KEYSET_SHA = "f2a75edba71b15928180dcd67103bbaea7f1d33c48be2bc219d762fcf95a99d2"
AUTHORITY_CHECKS_SHA = "e121d70e2b8f2d73bb00ba1d8dc6f22f76b6110c972e6cfbcf7454867b3d0fa9"
SYNTHETIC_EVIDENCE = {
    "passed": True,
    "check_count": 4,
    "checks": {
        "false_call_detected": True,
        "false_provenance_names_allowed": True,
        "literal_not_call": True,
        "two_argument_driver_accepted": True,
    },
    "fixture_sources_sha256": "77bc6af02c136be5b9b8c6bb653847de62311d5cd06b3547a06a60b2eee762fa",
    "evidence_sha256": "46a5fa5cfc6808b7fe52695248465cd694311b027b84f320e88a9ad7990d03a4",
}
PROOF_DIGESTS = {
    "determinism_scope_exact": "5bd6e5e1287440daa407d1099a193e464031e8dc4d76ac5ec1e56e88c4d3101c",
    "driver_owned_full_lifetime_logs_and_completion": "d00ff84e5f12b99ad89034e6734add7a4ab5b9760ab4e12ff0297528de862418",
    "no_training_reward_outcome": "3e49ce506211c09ad8b100118bf237b468dad2a2186f79cb6fc167c565092cce",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def regular(path: Path, digest: str, logical_bytes: int | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular canonical file: {path}")
    actual = {"path": str(path), "sha256": sha256_file(path), "logical_bytes": path.stat().st_size}
    if actual["sha256"] != digest or (logical_bytes is not None and actual["logical_bytes"] != logical_bytes):
        raise RuntimeError(f"file closure: {path}")
    return actual


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp):
        raise FileExistsError(path)
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exact_tree(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in rows).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }


def process_identity() -> dict:
    fields = Path("/proc/self/stat").read_text().split()
    return {
        "linux_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid": os.getpid(),
        "proc_self_stat_starttime_ticks": int(fields[21]),
        "sys_executable_resolved_path": str(Path(sys.executable).resolve()),
        "sys_executable_sha256": sha256_file(Path(sys.executable).resolve()),
    }


def no_live_process(executable_source: Path) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = [item.decode(errors="replace") for item in (entry / "cmdline").read_bytes().split(b"\0") if item]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(item == str(executable_source) for item in arguments):
            return False
    return True


def interpreter_closure() -> dict:
    lexical = Path(sys.executable)
    if lexical != RLPY_LEXICAL or not lexical.is_symlink() or lexical.resolve() != RLPY_RESOLVED:
        raise RuntimeError("RLPY lexical/resolved interpreter")
    resolved = regular(RLPY_RESOLVED, RLPY_SHA, RLPY_BYTES)
    versions = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
    }
    if versions != {"python": "3.11.15", "numpy": "1.26.4", "torch": "2.7.0+cu128"}:
        raise RuntimeError(f"RLPY versions: {versions}")
    return {"lexical_path": str(lexical), "lexical_is_symlink": True, "resolved": resolved, "versions": versions}


def record_from_payload(value: object, expected: dict, name: str) -> None:
    if value != expected:
        raise RuntimeError(f"authority record: {name}")


def preflight(args) -> dict:
    supplied_shas = (
        args.preregistration_sha,
        args.authority_sha,
        args.authority_design_contract_sha,
        args.reconciliation_design_contract_sha,
        args.phase_a_contract_sha,
        args.reconciler_sha,
        args.wrapper_sha,
    )
    if not all(HEX64.fullmatch(value or "") for value in supplied_shas):
        raise RuntimeError("placeholder or malformed SHA")
    exact_paths = {
        "preregistration": (args.preregistration, FORMAL_PATH, FORMAL_SHA, FORMAL_BYTES),
        "authority": (args.authority_receipt, AUTHORITY_PATH, args.authority_sha, None),
        "reconciliation_design_contract": (
            args.reconciliation_design_contract,
            RECON_CONTRACT_PATH,
            RECON_CONTRACT_SHA,
            RECON_CONTRACT_BYTES,
        ),
        "phase_a_contract": (args.phase_a_contract, PHASE_A_CONTRACT_PATH, PHASE_A_CONTRACT_SHA, None),
        "reconciler": (args.reconciler_source, RECONCILER_PATH, RECONCILER_SHA, RECONCILER_BYTES),
        "wrapper": (args.wrapper_source, Path(__file__).resolve(), args.wrapper_sha, None),
        "reconciliation_materializer": (MATERIALIZER_PATH, MATERIALIZER_PATH, MATERIALIZER_SHA, MATERIALIZER_BYTES),
        "old_preregistration": (OLD_FORMAL_PATH, OLD_FORMAL_PATH, OLD_FORMAL_SHA, 55761),
        "old_static_source": (OLD_STATIC_SOURCE, OLD_STATIC_SOURCE, OLD_STATIC_SHA, 29437),
        "old_static_receipt": (OLD_STATIC_RECEIPT, OLD_STATIC_RECEIPT, OLD_STATIC_RECEIPT_SHA, 7141),
        "old_static_log": (OLD_STATIC_LOG, OLD_STATIC_LOG, OLD_STATIC_LOG_SHA, 6475),
        "persistent_old_static_log": (
            PERSISTENT_OLD_STATIC_LOG,
            PERSISTENT_OLD_STATIC_LOG,
            OLD_STATIC_LOG_SHA,
            6475,
        ),
    }
    records = {}
    for name, (actual_path, expected_path, digest, size) in exact_paths.items():
        if actual_path.resolve() != expected_path or actual_path != expected_path:
            raise RuntimeError(f"exact path: {name}")
        records[name] = regular(actual_path, digest, size)
    if records["wrapper"]["sha256"] != args.wrapper_sha:
        raise RuntimeError("wrapper self SHA")
    authority_contract_record = regular(args.authority_design_contract, args.authority_design_contract_sha)
    if args.authority_design_contract != AUTHORITY_DESIGN_PATH or args.authority_design_contract.resolve() != AUTHORITY_DESIGN_PATH:
        raise RuntimeError("authority contract path")
    records["authority_design_contract"] = authority_contract_record
    formal = json.loads(args.preregistration.read_text())
    authority = json.loads(args.authority_receipt.read_text())
    authority_design = json.loads(args.authority_design_contract.read_text())
    if authority_design.get("format") != AUTHORITY_DESIGN_FORMAT or authority_design.get("status") != AUTHORITY_DESIGN_STATUS:
        raise RuntimeError("authority design boundary")
    authority_materializer_spec = authority_design.get("authority_materializer_source")
    if not isinstance(authority_materializer_spec, dict) or set(authority_materializer_spec) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("authority materializer design binding")
    records["authority_materializer"] = regular(
        Path(authority_materializer_spec["path"]),
        authority_materializer_spec["sha256"],
        authority_materializer_spec["logical_bytes"],
    )
    if records["authority_materializer"] != authority_materializer_spec or records["authority_materializer"] != {
        "path": str(AUTHORITY_MATERIALIZER_PATH),
        "sha256": AUTHORITY_MATERIALIZER_SHA,
        "logical_bytes": AUTHORITY_MATERIALIZER_BYTES,
    }:
        raise RuntimeError("authority materializer current closure")
    if formal.get("format") != "strict-track2-v486-v485-phase-a-static-false-positive-reconciliation-preregistration-v1":
        raise RuntimeError("formal format")
    if formal.get("status") != "preregistered_exact_one_readonly_static_reconciliation_authorized":
        raise RuntimeError("formal status")
    if formal.get("authorization") != AUTHORIZATION:
        raise RuntimeError("formal authorization")
    if formal.get("transparent_static_receipt_path") != str(TRANSPARENT_OUTPUT):
        raise RuntimeError("formal output")
    record_from_payload(formal.get("design_contract"), {"path": str(RECON_CONTRACT_PATH), "sha256": RECON_CONTRACT_SHA}, "formal contract")
    recon_record = formal.get("execution_sources", {}).get("reconciler")
    record_from_payload(recon_record, records["reconciler"], "formal reconciler")
    record_from_payload(formal.get("execution_sources", {}).get("reconciliation_materializer"), records["reconciliation_materializer"], "formal materializer")
    old_tree = exact_tree(OLD_FORMAL_PATH.parent)
    expected_old_tree = {
        "inventory": [["preregistration.json", OLD_FORMAL_SHA, 55761], ["static_audit.json", OLD_STATIC_RECEIPT_SHA, 7141]],
        "file_count": 2,
        "logical_file_bytes": 62902,
        "sha256sum_lines_digest_sha256": "f00ebe9d953b360b17fae68a2f32581a4add89bb90b86702be142c18b8e728df",
        "canonical_json_triples_digest_sha256": "199e880325f99e75b7d70e566d8da17c51093bafba3da1c3acd20d4a85c22c63",
    }
    if old_tree != expected_old_tree:
        raise RuntimeError("old b73 REG exact2")
    exact7 = formal.get("exact7_source_closure", {})
    exact7_records = exact7.get("records")
    if not isinstance(exact7_records, list) or len(exact7_records) != 7:
        raise RuntimeError("formal exact7 schema")
    observed_exact7 = []
    for entry in exact7_records:
        if set(entry) != {"role", "path", "sha256", "logical_bytes"}:
            raise RuntimeError("exact7 record schema")
        actual = regular(Path(entry["path"]), entry["sha256"], entry["logical_bytes"])
        observed_exact7.append({"role": entry["role"], **actual})
    if observed_exact7 != exact7_records or canonical_sha(observed_exact7) != exact7.get("canonical_records_digest_sha256"):
        raise RuntimeError("formal exact7 current closure")
    expected_reg = {
        "inventory": [
            ["immutable_evidence/v485_static_b73.log", OLD_STATIC_LOG_SHA, 6475],
            ["preregistration.json", FORMAL_SHA, FORMAL_BYTES],
        ],
        "file_count": 2,
        "logical_file_bytes": 27771,
        "sha256sum_lines_digest_sha256": "717d6ae12fbacbaafa147d03503140e5f807c028acc97a7294c84da2d79fe3fe",
        "canonical_json_triples_digest_sha256": "a0602666a7e59d60e464b3bc76b31b2fbf501e3d37bbd0b18b2db8c1233971ec",
    }
    if exact_tree(FORMAL_PATH.parent) != expected_reg:
        raise RuntimeError("f813 REG exact2")
    if exact_tree(AUTHORITY_ROOT) != {
        "inventory": [["authority_receipt.json", args.authority_sha, records["authority"]["logical_bytes"]]],
        "file_count": 1,
        "logical_file_bytes": records["authority"]["logical_bytes"],
        "sha256sum_lines_digest_sha256": hashlib.sha256(f"{args.authority_sha}  authority_receipt.json\n".encode()).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(
            json.dumps([["authority_receipt.json", args.authority_sha, records["authority"]["logical_bytes"]]], separators=(",", ":")).encode()
        ).hexdigest(),
    }:
        raise RuntimeError("authority REG exact1")
    if authority.get("format") != AUTH_FORMAT or authority.get("status") != AUTH_STATUS or authority.get("passed") is not True:
        raise RuntimeError("authority terminal")
    required_authority_keys = authority_design.get("authority_receipt_contract", {}).get("required_top_level_keys")
    if not isinstance(required_authority_keys, list) or set(authority) != set(required_authority_keys):
        raise RuntimeError("authority top-level schema")
    if authority.get("authorization") != AUTHORIZATION or authority.get("runtime_observation") != RUNTIME_PRE:
        raise RuntimeError("authority boundaries")
    record_from_payload(authority.get("authority_materializer_source"), records["authority_materializer"], "authority materializer")
    record_from_payload(authority.get("reconciliation_preregistration"), records["preregistration"], "authority formal")
    record_from_payload(authority.get("reconciliation_design_contract"), records["reconciliation_design_contract"], "authority contract")
    record_from_payload(
        authority.get("reconciliation_registration_materializer"),
        records["reconciliation_materializer"],
        "reconciliation materializer",
    )
    record_from_payload(authority.get("reconciler_source"), records["reconciler"], "authority reconciler")
    record_from_payload(authority.get("execution_wrapper_source"), records["wrapper"], "authority wrapper")
    record_from_payload(authority.get("authority_design_contract"), authority_contract_record, "authority design")
    if authority.get("attempt_root") != str(ATTEMPT_ROOT) or authority.get("transparent_static_receipt_path") != str(TRANSPARENT_OUTPUT):
        raise RuntimeError("authority destinations")
    if authority.get("old_b73_tree") != expected_old_tree or authority.get("reconciliation_REG_exact2") != expected_reg:
        raise RuntimeError("authority frozen trees")
    expected_old_records = {
        "preregistration": records["old_preregistration"],
        "phase_a_design_contract": records["phase_a_contract"],
        "old_static_source": records["old_static_source"],
        "failed_static_receipt": records["old_static_receipt"],
        "volatile_static_log": records["old_static_log"],
        "persistent_static_log": records["persistent_old_static_log"],
    }
    expected_old_ancestry = {
        "records": expected_old_records,
        "failed_checks": [
            "determinism_scope_exact",
            "driver_owned_full_lifetime_logs_and_completion",
            "no_training_reward_outcome",
        ],
        "true_check_count": 44,
    }
    if authority.get("old_parent_ancestry") != expected_old_ancestry:
        raise RuntimeError("authority old ancestry")
    if authority.get("exact7_source_closure") != {
        "records": exact7_records,
        "records_digest_sha256": SOURCE_RECORDS_SHA,
    }:
        raise RuntimeError("authority exact7 closure")
    transport = authority.get("transport_forensics")
    expected_transport = {
        **authority_design.get("materialization_transport_forensics", {}),
        "successful_materializer_invocation_persistent_argv_log_available": False,
        "successful_materializer_invocation_transport_disclosed": True,
    }
    if transport != expected_transport:
        raise RuntimeError("authority transport forensic disclosure")
    checks = authority.get("checks")
    if (
        checks != {key: True for key in AUTHORITY_CHECK_KEYS}
        or canonical_sha(sorted(checks)) != AUTHORITY_CHECK_KEYSET_SHA
        or canonical_sha(checks) != AUTHORITY_CHECKS_SHA
    ):
        raise RuntimeError("authority checks")
    if authority.get("input_pre_snapshot") != authority.get("input_post_snapshot") or authority.get("input_snapshots_exactly_equal") is not True:
        raise RuntimeError("authority snapshots")
    if "PENDING" in json.dumps(authority, sort_keys=True) or "PENDING" in json.dumps(authority_design, sort_keys=True):
        raise RuntimeError("authority placeholder")
    expected_historical_absences = {
        name: {**value, "absent": True}
        for name, value in authority_design.get("historical_pre_materialization_absences", {}).items()
    }
    historical_absences = authority.get("historical_pre_materialization_absences")
    if (
        not expected_historical_absences
        or historical_absences != expected_historical_absences
        or set(historical_absences) != {
            "qualification_root", "old_phase_a_launcher_attempts", "old_phase_a_authority_receipt",
            "transparent_static_receipt", "transparent_static_receipt_tmp", "reconciliation_attempt_root",
            "authority_root",
        }
        or Path(historical_absences["authority_root"]["path"]) != AUTHORITY_ROOT
    ):
        raise RuntimeError("authority historical absences")
    expected_required_absences = {
        name: {**value, "absent": True}
        for name, value in authority_design.get("required_current_absences_after_authority", {}).items()
    }
    required_absences = authority.get("required_absences")
    if (
        not expected_required_absences
        or required_absences != expected_required_absences
        or set(required_absences) != set(expected_historical_absences) - {"authority_root"}
    ):
        raise RuntimeError("authority absences")
    for value in required_absences.values():
        if os.path.lexists(value["path"]):
            raise RuntimeError("authority absence drift")
    own_absences = (
        ATTEMPT_ROOT,
        ATTEMPT_PREP,
        TRANSPARENT_OUTPUT,
        TRANSPARENT_OUTPUT.with_name(TRANSPARENT_OUTPUT.name + ".tmp"),
        QUALIFICATION_ROOT,
        OLD_FORMAL_PATH.parent / "phase_a_launcher_attempts",
        OLD_FORMAL_PATH.parent / "authority_receipt.json",
    )
    if any(os.path.lexists(path) for path in own_absences):
        raise RuntimeError("one-shot state exists")
    if not no_live_process(RECONCILER_PATH):
        raise RuntimeError("reconciler already running")
    if authority_design.get("future_execution_wrapper_source") != records["wrapper"]:
        raise RuntimeError("authority contract wrapper binding")
    return {
        "records": records,
        "authority_design_contract": authority_contract_record,
        "authority_checks_sha256": canonical_sha(checks),
        "f813_registration_tree": expected_reg,
        "authority_sha256": args.authority_sha,
        "old_b73_registration_tree": expected_old_tree,
        "exact7_source_records": observed_exact7,
        "exact7_source_records_sha256": canonical_sha(observed_exact7),
        "execution_interpreter": interpreter_closure(),
    }


def reconciler_command(args) -> list[str]:
    return [
        sys.executable,
        str(RECONCILER_PATH),
        "--preregistration", str(OLD_FORMAL_PATH), "--preregistration-sha", OLD_FORMAL_SHA,
        "--contract", str(PHASE_A_CONTRACT_PATH), "--contract-sha", PHASE_A_CONTRACT_SHA,
        "--old-static-source", str(OLD_STATIC_SOURCE), "--old-static-sha", OLD_STATIC_SHA,
        "--old-static-receipt", str(OLD_STATIC_RECEIPT), "--old-static-receipt-sha", OLD_STATIC_RECEIPT_SHA,
        "--old-static-log", str(OLD_STATIC_LOG), "--old-static-log-sha", OLD_STATIC_LOG_SHA,
        "--reconciliation-preregistration", str(FORMAL_PATH), "--reconciliation-preregistration-sha", FORMAL_SHA,
        "--reconciliation-contract", str(RECON_CONTRACT_PATH), "--reconciliation-contract-sha", RECON_CONTRACT_SHA,
        "--reconciler-source", str(RECONCILER_PATH), "--reconciler-sha", RECONCILER_SHA,
        "--output", str(TRANSPARENT_OUTPUT),
    ]


def acquire_execution_lock(path: Path) -> int:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def release_execution_lock(descriptor: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def owned_directory(path: Path, identity: tuple[int, int] | None) -> bool:
    if identity is None or not path.is_dir() or path.is_symlink():
        return False
    stat = path.stat()
    return (stat.st_dev, stat.st_ino) == identity


def commit_intent_at(attempt_root: Path, attempt_prep: Path, intent: dict) -> dict:
    prep_created_by_self = False
    prep_identity = None
    try:
        attempt_prep.mkdir()
        prep_created_by_self = True
        stat = attempt_prep.stat()
        prep_identity = (stat.st_dev, stat.st_ino)
        atomic_json(attempt_prep / "intent.json", intent)
        fsync_dir(attempt_prep)
        if not owned_directory(attempt_prep, prep_identity) or os.path.lexists(attempt_root):
            raise RuntimeError("attempt prep ownership before promote")
        os.replace(attempt_prep, attempt_root)
        fsync_dir(attempt_root.parent)
    except BaseException as error:
        if prep_created_by_self and owned_directory(attempt_prep, prep_identity) and not os.path.lexists(attempt_root):
            intent_path = attempt_prep / "intent.json"
            failure_path = attempt_prep / "terminal_receipt.json"
            if not os.path.lexists(failure_path) and not os.path.lexists(failure_path.with_name(failure_path.name + ".tmp")):
                failure = {
                    "format": TERMINAL_FORMAT,
                    "status": "failed_before_intent_promotion_no_retry",
                    "passed": False,
                    "attempt_nonce": intent["attempt_nonce"],
                    "intent_present": intent_path.is_file() and not intent_path.is_symlink(),
                    "intent_sha256": sha256_file(intent_path) if intent_path.is_file() and not intent_path.is_symlink() else None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **FALSE_AUTHORITIES,
                }
                atomic_json(failure_path, failure)
            fsync_dir(attempt_prep)
            if not owned_directory(attempt_prep, prep_identity) or os.path.lexists(attempt_root):
                raise RuntimeError("attempt prep ownership before failure promote") from error
            os.replace(attempt_prep, attempt_root)
            fsync_dir(attempt_root.parent)
        raise
    return regular(attempt_root / "intent.json", sha256_file(attempt_root / "intent.json"))


def commit_intent(intent: dict) -> dict:
    return commit_intent_at(ATTEMPT_ROOT, ATTEMPT_PREP, intent)


def terminate_group(process: subprocess.Popen | None) -> dict:
    result = {"term_sent": False, "kill_sent": False, "reaped": process is None, "process_group_empty": process is None}
    if process is None:
        return result
    try:
        os.killpg(process.pid, 0)
        group_live = True
    except ProcessLookupError:
        group_live = False
    if group_live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            result["term_sent"] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
        result["reaped"] = True
    except subprocess.TimeoutExpired:
        if group_live:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                result["kill_sent"] = True
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=30)
            result["reaped"] = True
        except subprocess.TimeoutExpired:
            result["reaped"] = False
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            group_live = False
            break
        time.sleep(0.05)
    if group_live:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            result["kill_sent"] = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                group_live = False
                break
            time.sleep(0.05)
    try:
        os.killpg(process.pid, 0)
        result["process_group_empty"] = False
    except ProcessLookupError:
        result["process_group_empty"] = True
    return result


def expected_reconciler_input_snapshot(closure: dict) -> dict:
    records = closure["records"]
    named = {
        "parent_preregistration": records["old_preregistration"],
        "phase_a_design_contract": records["phase_a_contract"],
        "old_static_source": records["old_static_source"],
        "old_static_receipt": records["old_static_receipt"],
        "volatile_old_static_log": records["old_static_log"],
        "reconciliation_preregistration": records["preregistration"],
        "reconciliation_design_contract": records["reconciliation_design_contract"],
        "reconciler_cli_source": records["reconciler"],
        "persistent_old_static_log": records["persistent_old_static_log"],
    }
    for record in closure["exact7_source_records"]:
        named[f"phase_a_exact7::{record['role']}"] = {
            key: record[key] for key in ("path", "sha256", "logical_bytes")
        }
    named["reconciliation_execution_source::reconciler"] = records["reconciler"]
    named["reconciliation_execution_source::reconciliation_materializer"] = records["reconciliation_materializer"]
    files = [{"label": label, **record} for label, record in sorted(named.items())]
    absences = [
        {"label": "old_phase_a_authority_receipt", "path": str((OLD_FORMAL_PATH.parent / "authority_receipt.json").resolve()), "absent": True},
        {"label": "old_phase_a_launcher_attempts", "path": str((OLD_FORMAL_PATH.parent / "phase_a_launcher_attempts").resolve()), "absent": True},
        {"label": "phase_a_qualification_root", "path": str(QUALIFICATION_ROOT.resolve()), "absent": True},
        {"label": "transparent_static_receipt", "path": str(TRANSPARENT_OUTPUT.resolve()), "absent": True},
        {"label": "transparent_static_receipt_tmp", "path": str(TRANSPARENT_OUTPUT.with_name(TRANSPARENT_OUTPUT.name + ".tmp").resolve()), "absent": True},
    ]
    payload = {
        "files": files,
        "trees": {
            "new_v486_initial_registration": closure["f813_registration_tree"],
            "old_b73_registration": closure["old_b73_registration_tree"],
        },
        "absences": absences,
    }
    return {**payload, "snapshot_sha256": canonical_sha(payload)}


def tree_with_appended_regular(before: dict, relative_path: str, record: dict) -> dict:
    if before.get("inventory") is None or any(row[0] == relative_path for row in before["inventory"]):
        raise RuntimeError("tree append collision")
    rows = sorted([*before["inventory"], [relative_path, record["sha256"], record["logical_bytes"]]])
    lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in rows).encode()
    return {
        "inventory": rows,
        "file_count": 3,
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }


def validate_transparent_check_evidence(value: dict, old_checks: dict) -> None:
    checks = value.get("checks")
    false_keys = {"determinism_scope_exact", "driver_owned_full_lifetime_logs_and_completion", "no_training_reward_outcome"}
    if (
        value.get("check_keys") != list(CHECK_KEYS)
        or canonical_sha(value.get("check_keys")) != CHECK_KEYSET_SHA
        or value.get("check_key_set_sha256") != CHECK_KEYSET_SHA
        or checks != {key: True for key in CHECK_KEYS}
        or value.get("checks_sha256") != ALL_TRUE_CHECKS_SHA
        or canonical_sha(checks) != ALL_TRUE_CHECKS_SHA
        or value.get("synthetic_recomputation_evidence") != SYNTHETIC_EVIDENCE
    ):
        raise RuntimeError("transparent exact check evidence")
    if (
        set(old_checks) != set(CHECK_KEYS)
        or {key for key, passed in old_checks.items() if passed is False} != false_keys
        or any(checks[key] is not True for key, passed in old_checks.items() if passed is True)
    ):
        raise RuntimeError("transparent old 44 plus corrected 3")


def validate_transparent(closure: dict) -> dict:
    record = regular(TRANSPARENT_OUTPUT, sha256_file(TRANSPARENT_OUTPUT))
    value = json.loads(TRANSPARENT_OUTPUT.read_text())
    expected_top_keys = {
        "format", "status", "passed", "reconciliation_format", "reconciliation_status", "checks", "check_keys",
        "check_key_set_sha256", "checks_sha256", "contract", "preregistration", "sources", "sources_digest_sha256",
        "static_auditor_self_sha256", "reconciliation_preregistration", "reconciliation_design_contract", "receipt_writer",
        "persistent_old_static_log", "volatile_log_source_at_registration", "registration_initial_inventory",
        "reconciliation_ancestry", "synthetic_recomputation_evidence", "input_pre_snapshot", "input_post_snapshot",
        "input_snapshots_exactly_equal", "false_positive_recomputation", "immutable_absences", "runtime_observation",
        "parent_runtime_observation", "reconciliation_runtime_observation", "phase_a_cache_qualification_authorized",
        "cache_reuse_authorized", "training_authorized", "folds_authorized", "policy_updates", "s1_authorized",
        "zero_update_authorized", "rl_authorized", "submission_authorized", "reward_read_authorized",
        "dev_hidden_final_outcome_read_authorized", "reconciliation_only", "phase_a_executed", "v169_imported_or_run",
    }
    if set(value) != expected_top_keys:
        raise RuntimeError("transparent top-level schema")
    if value.get("format") != "strict-track2-v485-v169-cache-qualification-static-audit-v1" or value.get("status") != "passed_no_execution_authority" or value.get("passed") is not True:
        raise RuntimeError("transparent terminal")
    if value.get("reconciliation_format") != "strict-track2-v486-v485-phase-a-transparent-static-reconciliation-v1" or value.get("reconciliation_status") != "passed_readonly_reconciliation_no_phase_a_execution_authority":
        raise RuntimeError("transparent reconciliation terminal")
    if value.get("reconciliation_preregistration") != {"path": str(FORMAL_PATH), "sha256": FORMAL_SHA}:
        raise RuntimeError("transparent formal")
    if value.get("reconciliation_design_contract") != {"path": str(RECON_CONTRACT_PATH), "sha256": RECON_CONTRACT_SHA}:
        raise RuntimeError("transparent contract")
    if value.get("receipt_writer") != {"path": str(RECONCILER_PATH), "sha256": RECONCILER_SHA, "logical_bytes": RECONCILER_BYTES}:
        raise RuntimeError("transparent writer")
    expected_snapshot = expected_reconciler_input_snapshot(closure)
    if value.get("input_pre_snapshot") != expected_snapshot or value.get("input_post_snapshot") != expected_snapshot or value.get("input_snapshots_exactly_equal") is not True:
        raise RuntimeError("transparent snapshots")
    old = json.loads(OLD_STATIC_RECEIPT.read_text())
    if (
        old.get("check_keys") != list(CHECK_KEYS)
        or old.get("check_key_set_sha256") != CHECK_KEYSET_SHA
        or old.get("checks_sha256") != canonical_sha(old.get("checks"))
    ):
        raise RuntimeError("transparent old check receipt")
    validate_transparent_check_evidence(value, old["checks"])
    checks = value["checks"]
    false_keys = {"determinism_scope_exact", "driver_owned_full_lifetime_logs_and_completion", "no_training_reward_outcome"}
    expected_sources = {
        record["role"]: {key: record[key] for key in ("path", "sha256", "logical_bytes")}
        for record in closure["exact7_source_records"]
    }
    if (
        value.get("contract") != {"path": str(PHASE_A_CONTRACT_PATH), "sha256": PHASE_A_CONTRACT_SHA}
        or value.get("preregistration") != {"path": str(OLD_FORMAL_PATH), "sha256": OLD_FORMAL_SHA}
        or value.get("sources") != expected_sources
        or value.get("sources_digest_sha256") != SOURCE_RECORDS_SHA
        or value.get("static_auditor_self_sha256") != OLD_STATIC_SHA
    ):
        raise RuntimeError("transparent parent source closure")
    expected_ancestry = {
        "old_static_source": closure["records"]["old_static_source"],
        "old_failed_static_receipt": closure["records"]["old_static_receipt"],
        "old_registration_tree": closure["old_b73_registration_tree"],
        "old_true_check_count": 44,
        "old_false_check_names": sorted(false_keys),
    }
    if value.get("reconciliation_ancestry") != expected_ancestry:
        raise RuntimeError("transparent reconciliation ancestry")
    if value.get("persistent_old_static_log") != closure["records"]["persistent_old_static_log"] or value.get("volatile_log_source_at_registration") != closure["records"]["old_static_log"]:
        raise RuntimeError("transparent log ancestry")
    if value.get("registration_initial_inventory") != closure["f813_registration_tree"]:
        raise RuntimeError("transparent initial REG")
    proofs = value.get("false_positive_recomputation")
    if not isinstance(proofs, dict) or set(proofs) != set(PROOF_DIGESTS):
        raise RuntimeError("transparent proof schema")
    for name, digest in PROOF_DIGESTS.items():
        if not isinstance(proofs[name], dict) or proofs[name].get("passed") is not True or canonical_sha(proofs[name]) != digest:
            raise RuntimeError(f"transparent proof: {name}")
    expected_immutable_absences = [
        str(QUALIFICATION_ROOT.resolve()),
        str((OLD_FORMAL_PATH.parent / "phase_a_launcher_attempts").resolve()),
        str((OLD_FORMAL_PATH.parent / "authority_receipt.json").resolve()),
    ]
    if value.get("immutable_absences") != expected_immutable_absences or any(os.path.lexists(path) for path in expected_immutable_absences):
        raise RuntimeError("transparent immutable absences")
    if any(value.get(key) != expected for key, expected in FALSE_AUTHORITIES.items()):
        raise RuntimeError("transparent authority")
    parent_runtime = {"phase_a_executed": False, "training_launched": False, "folds": 0, "policy_updates": 0, "rl_authorized": False}
    if value.get("runtime_observation") != parent_runtime or value.get("parent_runtime_observation") != parent_runtime or value.get("reconciliation_only") is not True or value.get("phase_a_executed") is not False or value.get("v169_imported_or_run") is not False:
        raise RuntimeError("transparent compatibility runtime")
    runtime = value.get("reconciliation_runtime_observation", {})
    expected_runtime = {"reconciliation_executed": True, "phase_a_executed": False, "v169_imported_or_run": False, "qualification_root_created": False, "training_launched": False, "folds": 0, "policy_updates": 0, "reward_read_or_loaded": False, "dev_hidden_final_outcome_read": False}
    if runtime != expected_runtime or os.path.lexists(QUALIFICATION_ROOT):
        raise RuntimeError("transparent runtime")
    post_tree = exact_tree(FORMAL_PATH.parent)
    expected_post_tree = tree_with_appended_regular(
        closure["f813_registration_tree"], "transparent_static_audit.json", record
    )
    if post_tree != expected_post_tree or os.path.lexists(TRANSPARENT_OUTPUT.with_name(TRANSPARENT_OUTPUT.name + ".tmp")):
        raise RuntimeError("transparent sole REG transition")
    return {**record, "checks_sha256": ALL_TRUE_CHECKS_SHA, "post_registration_tree": post_tree}


def synthetic_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="v486-wrapper-") as directory:
        root = Path(directory)
        atomic_root = root / "atomic"
        atomic_root.mkdir()
        payload = {"a": 1, "passed": False}
        atomic_json(atomic_root / "x.json", payload)
        tree = exact_tree(atomic_root)
        checks = {
            "atomic_json_one_file": tree["file_count"] == 1,
            "atomic_json_roundtrip": json.loads((atomic_root / "x.json").read_text()) == payload,
            "canonical_digest_stable": canonical_sha(payload) == canonical_sha(dict(payload)),
            "false_authorities_exact": FALSE_AUTHORITIES["training_authorized"] is False and FALSE_AUTHORITIES["policy_updates"] == 0,
        }
        loser_root = root / "loser-final"
        loser_prep = root / "loser-prep"
        loser_prep.mkdir()
        (loser_prep / "foreign.marker").write_text("foreign")
        loser_rejected = False
        try:
            commit_intent_at(loser_root, loser_prep, {"attempt_nonce": "0" * 64})
        except FileExistsError:
            loser_rejected = True
        checks["foreign_prep_loser_never_touched"] = (
            loser_rejected
            and (loser_prep / "foreign.marker").read_text() == "foreign"
            and not os.path.lexists(loser_root)
            and not os.path.lexists(loser_prep / "terminal_receipt.json")
        )
        owner_root = root / "owner-final"
        owner_prep = root / "owner-prep"
        owner_intent = {"attempt_nonce": "1" * 64, "passed": False}
        owner_record = commit_intent_at(owner_root, owner_prep, owner_intent)
        checks["self_owned_prep_atomic_promote"] = (
            owner_record["path"] == str((owner_root / "intent.json").resolve())
            and json.loads((owner_root / "intent.json").read_text()) == owner_intent
            and not os.path.lexists(owner_prep)
        )
        old_checks = {key: True for key in CHECK_KEYS}
        for key in ("determinism_scope_exact", "driver_owned_full_lifetime_logs_and_completion", "no_training_reward_outcome"):
            old_checks[key] = False
        exact_evidence = {
            "check_keys": list(CHECK_KEYS),
            "check_key_set_sha256": CHECK_KEYSET_SHA,
            "checks": {key: True for key in CHECK_KEYS},
            "checks_sha256": ALL_TRUE_CHECKS_SHA,
            "synthetic_recomputation_evidence": SYNTHETIC_EVIDENCE,
        }
        validate_transparent_check_evidence(exact_evidence, old_checks)
        tampered = json.loads(json.dumps(exact_evidence))
        tampered["checks"]["determinism_scope_exact"] = False
        tamper_rejected = False
        try:
            validate_transparent_check_evidence(tampered, old_checks)
        except RuntimeError:
            tamper_rejected = True
        checks["transparent_exact47_tamper_rejected"] = tamper_rejected
        post_root = root / "post-tree"
        post_root.mkdir()
        (post_root / "a").write_bytes(b"a")
        (post_root / "b").write_bytes(b"bb")
        before = exact_tree(post_root)
        (post_root / "c").write_bytes(b"ccc")
        c_record = regular(post_root / "c", sha256_file(post_root / "c"))
        expected_after = tree_with_appended_regular(before, "c", c_record)
        checks["registration_exact2_to_exact3"] = exact_tree(post_root) == expected_after and expected_after["file_count"] == 3
        (post_root / "extra").write_bytes(b"forbidden")
        checks["registration_extra_rejected"] = exact_tree(post_root) != expected_after
        if os.name == "posix":
            checks["rlpy_interpreter_closure"] = interpreter_closure()["resolved"]["sha256"] == RLPY_SHA
            process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            cleanup = terminate_group(process)
            checks["bounded_process_group_cleanup"] = cleanup["reaped"] is True and cleanup["process_group_empty"] is True
            lock_path = root / "race.lock"
            lock_path.write_bytes(b"")
            lock_fd = acquire_execution_lock(lock_path)
            contender = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import fcntl,os,sys;fd=os.open(sys.argv[1],os.O_RDONLY);"
                    "\ntry: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);sys.exit(1)"
                    "\nexcept BlockingIOError: sys.exit(0)",
                    str(lock_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            release_execution_lock(lock_fd)
            checks["external_flock_rejects_racing_contender"] = contender.returncode == 0
        else:
            checks["rlpy_interpreter_closure"] = True
            checks["bounded_process_group_cleanup"] = True
            checks["external_flock_rejects_racing_contender"] = True
    print(json.dumps({"passed": set(checks.values()) == {True}, "checks": checks}, sort_keys=True))
    return 0 if set(checks.values()) == {True} else 3


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--authority-receipt", type=Path, required=True)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--authority-design-contract", type=Path, required=True)
    parser.add_argument("--authority-design-contract-sha", required=True)
    parser.add_argument("--reconciliation-design-contract", type=Path, required=True)
    parser.add_argument("--reconciliation-design-contract-sha", required=True)
    parser.add_argument("--phase-a-contract", type=Path, required=True)
    parser.add_argument("--phase-a-contract-sha", required=True)
    parser.add_argument("--reconciler-source", type=Path, required=True)
    parser.add_argument("--reconciler-sha", required=True)
    parser.add_argument("--wrapper-source", type=Path, required=True)
    parser.add_argument("--wrapper-sha", required=True)
    args = parser.parse_args()
    if args.preregistration_sha != FORMAL_SHA or args.reconciliation_design_contract_sha != RECON_CONTRACT_SHA or args.phase_a_contract_sha != PHASE_A_CONTRACT_SHA or args.reconciler_sha != RECONCILER_SHA:
        raise RuntimeError("frozen argument SHA")
    # First pass is strictly read-only.  The authority receipt itself is then
    # flocked for the lifetime of this one-shot process, and every input is
    # revalidated under that lock before the first attempt mutation.
    closure = preflight(args)
    _execution_lock_fd = acquire_execution_lock(AUTHORITY_PATH)
    closure = preflight(args)
    command = reconciler_command(args)
    intent = {
        "format": INTENT_FORMAT,
        "status": "committed_before_reconciler_start",
        "attempt_nonce": os.urandom(32).hex(),
        "created_epoch_ns": time.time_ns(),
        "process_identity": process_identity(),
        "command_argv": command,
        "command_argv_sha256": canonical_sha(command),
        "closure": closure,
        "transparent_output_path": str(TRANSPARENT_OUTPUT),
        "timeout_seconds": 300,
        **FALSE_AUTHORITIES,
    }
    stdout_path = ATTEMPT_ROOT / "reconciler_stdout.log"
    stderr_path = ATTEMPT_ROOT / "reconciler_stderr.log"
    process = None
    started = time.monotonic_ns()
    terminal_committed = False
    intent_record = None
    def interrupted(signum, _frame):
        raise InterruptedError(f"launcher signal {signum}")
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        intent_record = commit_intent(intent)
        with stdout_path.open("xb", buffering=0) as stdout, stderr_path.open("xb", buffering=0) as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                returncode = process.wait(timeout=300)
            except subprocess.TimeoutExpired as error:
                cleanup = terminate_group(process)
                raise RuntimeError({"stage": "reconciler_timeout", "cleanup": cleanup}) from error
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        cleanup = terminate_group(process)
        if returncode != 0 or cleanup["reaped"] is not True or cleanup["process_group_empty"] is not True:
            raise RuntimeError({"stage": "reconciler_nonzero_or_not_empty", "returncode": returncode, "cleanup": cleanup})
        transparent = validate_transparent(closure)
        post_registration_tree = transparent["post_registration_tree"]
        registration_transition_sha256 = canonical_sha(
            {"before": closure["f813_registration_tree"], "after": post_registration_tree}
        )
        terminal = {
            "format": TERMINAL_FORMAT,
            "status": "passed_readonly_reconciliation_no_phase_a_execution_authority",
            "passed": True,
            "wall_seconds": float((time.monotonic_ns() - started) / 1e9),
            "intent": intent_record,
            "reconciler_exit_code": 0,
            "reconciler_process_group_empty": True,
            "reconciler_stdout": regular(stdout_path, sha256_file(stdout_path)),
            "reconciler_stderr": regular(stderr_path, sha256_file(stderr_path)),
            "transparent_static_receipt": transparent,
            "qualification_root_absent": not os.path.lexists(QUALIFICATION_ROOT),
            "f813_registration_tree_before": closure["f813_registration_tree"],
            "f813_registration_tree_after": post_registration_tree,
            "f813_registration_tree_transition_sha256": registration_transition_sha256,
            **FALSE_AUTHORITIES,
        }
        if not math.isfinite(terminal["wall_seconds"]) or terminal["wall_seconds"] < 0:
            raise RuntimeError("terminal wall")
        atomic_json(ATTEMPT_ROOT / "terminal_receipt.json", terminal)
        terminal_committed = True
        return 0
    except BaseException as error:
        cleanup = terminate_group(process)
        if terminal_committed or (ATTEMPT_ROOT / "terminal_receipt.json").is_file():
            raise
        if not ATTEMPT_ROOT.is_dir() or ATTEMPT_ROOT.is_symlink():
            raise
        if intent_record is None:
            intent_path = ATTEMPT_ROOT / "intent.json"
            intent_record = regular(intent_path, sha256_file(intent_path)) if intent_path.is_file() and not intent_path.is_symlink() else None
        for path in (stdout_path, stderr_path):
            if path.is_file() and not path.is_symlink():
                descriptor = os.open(str(path), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)
        failure = {
            "format": TERMINAL_FORMAT,
            "status": "failed_no_retry",
            "passed": False,
            "wall_seconds": float((time.monotonic_ns() - started) / 1e9),
            "intent": intent_record,
            "error_type": type(error).__name__,
            "error": str(error),
            "cleanup": cleanup,
            "reconciler_stdout": regular(stdout_path, sha256_file(stdout_path)) if stdout_path.is_file() else None,
            "reconciler_stderr": regular(stderr_path, sha256_file(stderr_path)) if stderr_path.is_file() else None,
            "transparent_output_present": TRANSPARENT_OUTPUT.is_file() and not TRANSPARENT_OUTPUT.is_symlink(),
            "qualification_root_absent": not os.path.lexists(QUALIFICATION_ROOT),
            **FALSE_AUTHORITIES,
        }
        atomic_json(ATTEMPT_ROOT / "terminal_receipt.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
