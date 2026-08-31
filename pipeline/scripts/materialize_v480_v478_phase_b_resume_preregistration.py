#!/usr/bin/env python3
"""Register immutable batch 0 and freeze the v478 Phase-B resume at batch 1."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
S = ROOT / "pipeline/scripts"
OLD_REG = J / "v478_temporal200_seed1622_20260824"
V480 = J / "v480_v478_phase_b_batch0_reconciliation_20260824"
NEW_REG = J / "v480_v478_phase_b_resume_seed1622_20260824"
STAGING_REG = J / ".v480_v478_phase_b_resume_seed1622_20260824.registration-prep"
DATASET = Path("/root/v478_temporal8_dataset_seed1622_20260824")
OLD_ATTEMPT = OLD_REG / "phase_b_launcher_attempts/20260823T234034-277910"
AUDIT_DIR = NEW_REG / "phase_b_batch_audits"
REGISTERED_AUDIT = AUDIT_DIR / "batch_000_audit.json"
OUTPUT = NEW_REG / "phase_b_resume_preregistration.json"
STAGING_AUDIT_DIR = STAGING_REG / "phase_b_batch_audits"
STAGING_AUDIT = STAGING_AUDIT_DIR / "batch_000_audit.json"
STAGING_OUTPUT = STAGING_REG / "phase_b_resume_preregistration.json"

PARENT_FORMAT = "strict-track2-v478-public-train-temporal200-preregistration-v1"
PARENT_STATUS = "preregistered_public_train_temporal_collection_authorized"
WHOLE_WALL = 43200.0
PRIOR_DATASET_WALL = 3214.017167359125
PRIOR_LAUNCHER_WALL = 3231.3513736724854


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or sha(resolved) != expected:
        raise RuntimeError(f"immutable closure drift: {resolved}")
    return resolved


def inventory(root: Path) -> tuple[list[list[object]], str, str]:
    paths = list(root.resolve().rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("batch0 symlink forbidden")
    items = [
        [path.relative_to(root).as_posix(), path.stat().st_size, sha(path)]
        for path in sorted(paths)
        if path.is_file()
    ]
    compact = json.dumps([[item[0], item[2]] for item in items], sort_keys=True, separators=(",", ":"))
    lines = "".join(f"{item[2]}  {item[0]}\n" for item in items)
    return items, hashlib.sha256(compact.encode()).hexdigest(), hashlib.sha256(lines.encode()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())


def record(path: Path, expected: str) -> dict[str, str]:
    resolved = require(path, expected)
    return {"path": str(resolved), "sha256": expected}


def main() -> int:
    if NEW_REG.exists() or STAGING_REG.exists():
        raise FileExistsError(NEW_REG if NEW_REG.exists() else STAGING_REG)

    parent_path = require(
        OLD_REG / "phase_b_preregistration.json",
        "5d6db6e5aca4502b79347620a7732e4471f63b77780ad7228bcd53c6cb753ad9",
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("format") != PARENT_FORMAT or parent.get("status") != PARENT_STATUS:
        raise RuntimeError("parent preregistration drift")

    fixed_auditor = require(
        S / "audit_v480_v478_temporal200_sharded.py",
        "0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0",
    )
    generator = require(
        S / "generate_v478_temporal200_sharded.py",
        "a20f5e9d702fa6febe946d9f713f4713ef035705b35119c00ca328046da70a58",
    )
    collector = require(
        S / "collect_v477_temporal_paired.py",
        "be90f3dd1272706af6d9bc1f461543c9b399071139a4ea243bd962b4b01f187a",
    )
    reconciliation_contract = require(
        S / "v480_v478_batch0_immutable_reconciliation_contract.json",
        "2a492ddd0b974a74c5cbd8f1d11b093905c74f931da76abd745d94fea6aec827",
    )
    materializer = Path(__file__).resolve()

    v480_pre = record(
        V480 / "preregistration.json",
        "40d6116fff8d42dc279f543b7a5be9d995051ed8264e92b984ea2b9be5487536",
    )
    v480_audit = record(
        V480 / "recomputed_batch_000_audit.json",
        "713c6976a42e817d1f66414b30b35c6e72f505a2a4f68e9a4b778adc8bca04b9",
    )
    v480_reconciliation = record(
        V480 / "reconciliation_receipt.json",
        "4cc7d26eccf4b36a4fee4fe7f7f8ad23426b16fd1791cd37988745edca4eeed1",
    )
    v480_launcher = record(
        V480 / "launcher_receipt.json",
        "332873cba84cea5b5f5e4c7d9576a24f09292b6a2b863d985219b4cdc3965558",
    )
    v480_pre_payload = json.loads(Path(v480_pre["path"]).read_text(encoding="utf-8"))
    v480_evidence_rows = []
    for key, item in sorted(v480_pre_payload.get("immutable_evidence", {}).items()):
        evidence_path = require(Path(item["path"]), item["sha256"])
        v480_evidence_rows.append([key, str(evidence_path), item["sha256"]])
    if not v480_evidence_rows or any(Path(path).exists() for path in v480_pre_payload.get("required_absence", [])):
        raise RuntimeError("v480 frozen evidence or required absence drift")
    v480_evidence_digest = hashlib.sha256(
        json.dumps(v480_evidence_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    old_failure = record(
        OLD_ATTEMPT / "failure_receipt.json",
        "48c95faeda8992ebf041d677a1b3e231b4619269f5e86574c3f8a73473ac5452",
    )
    batch_report_path = require(
        DATASET / "batch_000/batch_report.json",
        "59f77e4f059732ea49bd922bf74551cd93d05248ebd121fc6741e645eb02f7dc",
    )

    reconciliation = json.loads(Path(v480_reconciliation["path"]).read_text(encoding="utf-8"))
    launcher = json.loads(Path(v480_launcher["path"]).read_text(encoding="utf-8"))
    audit = json.loads(Path(v480_audit["path"]).read_text(encoding="utf-8"))
    report = json.loads(batch_report_path.read_text(encoding="utf-8"))
    failure = json.loads(Path(old_failure["path"]).read_text(encoding="utf-8"))
    expected_authorization = {
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
    }
    if reconciliation.get("passed") is not True or reconciliation.get("authorization") != expected_authorization:
        raise RuntimeError("v480 reconciliation authorization drift")
    if launcher.get("passed") is not True or any(launcher.get(key) != value for key, value in expected_authorization.items()):
        raise RuntimeError("v480 launcher authorization drift")
    if audit.get("passed") is not True or len(audit.get("checks", {})) != 7 or not all(audit["checks"].values()):
        raise RuntimeError("v480 batch0 audit drift")
    if report.get("passed") is not True or float(report.get("cumulative_attempt_wall_seconds")) != PRIOR_DATASET_WALL:
        raise RuntimeError("batch0 dataset wall drift")
    if float(failure.get("cumulative_launcher_wall_seconds")) != PRIOR_LAUNCHER_WALL:
        raise RuntimeError("parent launcher wall drift")

    items, compact_digest, lines_digest = inventory(DATASET / "batch_000")
    if not (
        len(items) == 41
        and sum(int(item[1]) for item in items) == 201004510
        and compact_digest == "0ff88bdbd6196d33d94796d7733fdc81bd2102a55a8c53d611c1a9454414d1b5"
        and lines_digest == "b9a44ba8b63c2cc721e6bd94d03d38efff61f18fa32fe437904faf88e3e2d031"
    ):
        raise RuntimeError("batch0 immutable tree drift")
    for forbidden in (
        DATASET / "batch_001",
        DATASET / "batch_001.partial",
        DATASET / "generation_report.json",
        OLD_REG / "phase_b_batch_audits/batch_000_audit.json",
        OLD_REG / "phase_b_final_audit.json",
        OLD_REG / "phase_b_launcher_receipt.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"unexpected resume/final artifact: {forbidden}")

    STAGING_REG.mkdir(parents=False, exist_ok=False)
    STAGING_AUDIT_DIR.mkdir(parents=False, exist_ok=False)
    atomic_bytes(STAGING_AUDIT, Path(v480_audit["path"]).read_bytes())
    if sha(STAGING_AUDIT) != v480_audit["sha256"]:
        raise RuntimeError("registered batch0 audit copy drift")

    payload = copy.deepcopy(parent)
    payload["registry_root"] = str(NEW_REG.resolve())
    payload["execution_closure"]["auditor"] = {
        "path": str(fixed_auditor),
        "sha256": "0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0",
    }
    payload["execution_closure"].update(
        {
            "original_phase_b_preregistration": {"path": str(parent_path), "sha256": sha(parent_path)},
            "v480_reconciliation_preregistration": v480_pre,
            "v480_recomputed_batch0_audit": v480_audit,
            "v480_reconciliation_receipt": v480_reconciliation,
            "v480_reconciliation_launcher_receipt": v480_launcher,
            "v480_reconciliation_contract": {
                "path": str(reconciliation_contract),
                "sha256": sha(reconciliation_contract),
            },
            "v480_resume_materializer": {"path": str(materializer), "sha256": sha(materializer)},
            "parent_launcher_failure_receipt": old_failure,
            "registered_batch0_audit": {"path": str(REGISTERED_AUDIT.resolve()), "sha256": v480_audit["sha256"]},
        }
    )
    if payload["execution_closure"]["generator"]["sha256"] != sha(generator) or payload["execution_closure"]["collector"]["sha256"] != sha(collector):
        raise RuntimeError("parent generator or collector drift")
    payload["lineage"] = {
        "format": "strict-track2-v480-v478-phase-b-resume-lineage-v1",
        "parent_preregistration_sha256": sha(parent_path),
        "batch0_reconciliation_preregistration_sha256": v480_pre["sha256"],
        "batch0_reconciliation_receipt_sha256": v480_reconciliation["sha256"],
        "batch0_reconciliation_launcher_receipt_sha256": v480_launcher["sha256"],
        "no_retry_of_failed_parent_attempt": True,
        "v480_immutable_evidence_closure": {
            "canonical_compact_json_key_path_sha256_rows_digest": v480_evidence_digest,
            "record_count": len(v480_evidence_rows),
            "all_paths_and_sha256_reverified": True,
            "all_v480_required_absences_reverified": True
        },
    }
    payload["resume_registration"] = {
        "format": "strict-track2-v480-v478-phase-b-resume-registration-v1",
        "authorized": True,
        "next_batch_id": 1,
        "batch0_reuse_required": True,
        "batch0_rerun_authorized": False,
        "batch0_generator_invocations_allowed": 0,
        "registered_batch0_audit": {"path": str(REGISTERED_AUDIT.resolve()), "sha256": v480_audit["sha256"]},
        "batch0_tree": {
            "file_count": 41,
            "logical_regular_file_bytes": 201004510,
            "canonical_compact_json_relative_sha256_pairs_sha256": compact_digest,
            "sorted_sha256_double_space_relative_posix_lines_sha256": lines_digest,
        },
        "effect_or_outcome_conditioned_retry": False,
    }
    payload["wall_accounting"] = {
        "whole_wall_seconds_max": WHOLE_WALL,
        "prior_dataset_cumulative_attempt_wall_seconds": PRIOR_DATASET_WALL,
        "remaining_dataset_attempt_wall_seconds": WHOLE_WALL - PRIOR_DATASET_WALL,
        "prior_launcher_cumulative_wall_seconds": PRIOR_LAUNCHER_WALL,
        "remaining_launcher_wall_seconds": WHOLE_WALL - PRIOR_LAUNCHER_WALL,
        "launcher_watchdog_prior_wall_seconds_ceil": 3232,
        "dataset_and_launcher_accounting_are_distinct": True,
        "accounting_difference_seconds": PRIOR_LAUNCHER_WALL - PRIOR_DATASET_WALL,
        "v480_reconciliation_excluded_from_collection_wall": True,
    }
    payload["guards"].update(
        {
            "v480_batch0_reconciliation_bound": True,
            "next_batch_id": 1,
            "batch0_reuse_required": True,
            "batch0_rerun_authorized": False,
            "batch0_generator_invocations_allowed": 0,
            "effect_or_outcome_conditioned_retry": False,
            "training_authorized": False,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        }
    )
    atomic_json(STAGING_OUTPUT, payload)
    for directory in (STAGING_AUDIT_DIR, STAGING_REG):
        descriptor = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.replace(STAGING_REG, NEW_REG)
    descriptor = os.open(str(J), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if sha(REGISTERED_AUDIT) != v480_audit["sha256"]:
        raise RuntimeError("promoted registered batch0 audit drift")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha(OUTPUT), "registered_batch0_audit_sha256": sha(REGISTERED_AUDIT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
