#!/usr/bin/env python3
"""Freeze the preflight-only v480 resume failure and register resume-r2 at batch 1."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
S = ROOT / "pipeline/scripts"
DATASET = Path("/root/v478_temporal8_dataset_seed1622_20260824")
PARENT_REG = J / "v480_v478_phase_b_resume_seed1622_20260824"
FAILED_ATTEMPT = PARENT_REG / "phase_b_launcher_attempts/20260824T011241-383654"
NEW_REG = J / "v481_v480_phase_b_resume_r2_seed1622_20260824"
STAGING_REG = J / ".v481_v480_phase_b_resume_r2_seed1622_20260824.registration-prep"
PARENT_FORMAL = PARENT_REG / "phase_b_resume_preregistration.json"
PARENT_AUDIT = PARENT_REG / "phase_b_batch_audits/batch_000_audit.json"
STAGING_AUDIT_DIR = STAGING_REG / "phase_b_batch_audits"
STAGING_AUDIT = STAGING_AUDIT_DIR / "batch_000_audit.json"
FINAL_AUDIT = NEW_REG / "phase_b_batch_audits/batch_000_audit.json"
STAGING_EVIDENCE_DIR = STAGING_REG / "immutable_evidence"
STAGING_OUTER_LOG = STAGING_EVIDENCE_DIR / "v480_failed_resume_outer.log"
FINAL_OUTER_LOG = NEW_REG / "immutable_evidence/v480_failed_resume_outer.log"
STAGING_FORMAL = STAGING_REG / "phase_b_resume_preregistration.json"
FINAL_FORMAL = NEW_REG / "phase_b_resume_preregistration.json"
VOLATILE_OUTER_LOG = Path("/tmp/v480_phase_b_resume_outer.log")

FORMAT = "strict-track2-v478-public-train-temporal200-preregistration-v1"
STATUS = "preregistered_public_train_temporal_collection_authorized"
WHOLE_WALL = 43200.0
DATASET_PRIOR = 3214.017167359125
ORIGINAL_LAUNCHER_RAW = 3231.3513736724854
ORIGINAL_LAUNCHER_CEIL = 3232
FAILED_ATTEMPT_WALL = 5.613072872161865
FAILED_CUMULATIVE_RAW = 3237.613072872162
NEXT_WATCHDOG_CEIL = 3238


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or sha(resolved) != expected:
        raise RuntimeError(f"immutable evidence drift: {resolved}")
    return resolved


def record(path: Path, expected: str) -> dict[str, str]:
    resolved = require(path, expected)
    return {"path": str(resolved), "sha256": expected}


def inventory(root: Path) -> tuple[list[list[object]], str, str]:
    paths = list(root.resolve().rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("batch0 symlink forbidden")
    items = [
        [path.relative_to(root).as_posix(), path.stat().st_size, sha(path)]
        for path in sorted(paths)
        if path.is_file()
    ]
    pairs = json.dumps([[item[0], item[2]] for item in items], sort_keys=True, separators=(",", ":"))
    lines = "".join(f"{item[2]}  {item[0]}\n" for item in items)
    return items, hashlib.sha256(pairs.encode()).hexdigest(), hashlib.sha256(lines.encode()).hexdigest()


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


def main() -> int:
    if NEW_REG.exists() or STAGING_REG.exists():
        raise FileExistsError(NEW_REG if NEW_REG.exists() else STAGING_REG)

    parent_path = require(PARENT_FORMAL, "a1614185001ab98981bd394b76fac6b7220bd9e6878fff1098a712750c9a8f79")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("format") != FORMAT or parent.get("status") != STATUS:
        raise RuntimeError("parent resume formal drift")
    old_wall = parent.get("wall_accounting", {})
    expected_old_wall = {
        "whole_wall_seconds_max": WHOLE_WALL,
        "prior_dataset_cumulative_attempt_wall_seconds": DATASET_PRIOR,
        "remaining_dataset_attempt_wall_seconds": WHOLE_WALL - DATASET_PRIOR,
        "prior_launcher_cumulative_wall_seconds": ORIGINAL_LAUNCHER_RAW,
        "remaining_launcher_wall_seconds": WHOLE_WALL - ORIGINAL_LAUNCHER_RAW,
        "launcher_watchdog_prior_wall_seconds_ceil": ORIGINAL_LAUNCHER_CEIL,
        "dataset_and_launcher_accounting_are_distinct": True,
        "accounting_difference_seconds": ORIGINAL_LAUNCHER_RAW - DATASET_PRIOR,
        "v480_reconciliation_excluded_from_collection_wall": True,
    }
    for key, expected in expected_old_wall.items():
        actual = old_wall.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"parent wall formula drift: {key}")
        elif actual != expected:
            raise RuntimeError(f"parent wall field drift: {key}")

    parent_audit = record(PARENT_AUDIT, "713c6976a42e817d1f66414b30b35c6e72f505a2a4f68e9a4b778adc8bca04b9")
    old_launcher = record(
        S / "launch_v480_v478_temporal200_phase_b_resume.sh",
        "9775ad899b1a2009e574175a807a31298d11d8024eab7450ce8d207c65737d60",
    )
    attempt_intent = record(
        FAILED_ATTEMPT / "attempt_intent.json",
        "b4df508bf1899bd027755739c5b522c0bdfc11a8e98ffa293bc88845d4be0d22",
    )
    failure_receipt = record(
        FAILED_ATTEMPT / "failure_receipt.json",
        "7b580c67ea7ed4f5d6bdbc9c97eb6a5f6aecf0fbda1505fed9f0c8ea05fe120b",
    )
    restore_log = record(
        FAILED_ATTEMPT / "restore_v218_00.log",
        "bf36c8bfe0d2f0af1690b20840a80e622e67e3029daab4b4985a88fc2989acb1",
    )
    outer_log = require(
        VOLATILE_OUTER_LOG,
        "bf53059c107ba95f4d2c9716876f1f50941290d5a0b05def202eadfdbc55269d",
    )
    materializer = Path(__file__).resolve()

    intent = json.loads(Path(attempt_intent["path"]).read_text(encoding="utf-8"))
    failure = json.loads(Path(failure_receipt["path"]).read_text(encoding="utf-8"))
    if not (
        intent.get("preregistration_sha256") == sha(parent_path)
        and intent.get("launcher_sha256") == old_launcher["sha256"]
        and failure.get("passed") is False
        and failure.get("stage") == "preflight_restore1"
        and failure.get("batch_id") == -1
        and failure.get("exit_code") == 1
        and failure.get("launcher_sha256") == old_launcher["sha256"]
        and failure.get("preregistration_sha256") == sha(parent_path)
        and failure.get("v218_health_restored") is True
        and failure.get("generation_report") == {"exists": False, "sha256": None}
        and failure.get("final_audit") == {"exists": False, "sha256": None}
        and failure.get("training_authorized") is False
        and failure.get("s1_authorized") is False
        and failure.get("zero_update_authorized") is False
        and failure.get("policy_updates") == 0
        and failure.get("rl_authorized") is False
        and math.isclose(float(failure.get("launcher_wall_seconds")), FAILED_ATTEMPT_WALL, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(failure.get("cumulative_launcher_wall_seconds")), FAILED_CUMULATIVE_RAW, rel_tol=0.0, abs_tol=1e-12)
        and int(failure.get("prior_launcher_wall_seconds")) == ORIGINAL_LAUNCHER_CEIL
    ):
        raise RuntimeError("failed resume evidence semantic drift")
    if "AssertionError" not in outer_log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("outer preflight failure evidence drift")

    items, compact_digest, lines_digest = inventory(DATASET / "batch_000")
    if not (
        len(items) == 41
        and sum(int(item[1]) for item in items) == 201004510
        and compact_digest == "0ff88bdbd6196d33d94796d7733fdc81bd2102a55a8c53d611c1a9454414d1b5"
        and lines_digest == "b9a44ba8b63c2cc721e6bd94d03d38efff61f18fa32fe437904faf88e3e2d031"
    ):
        raise RuntimeError("batch0 tree drift")
    for forbidden in (
        DATASET / "batch_001",
        DATASET / "batch_001.partial",
        DATASET / "generation_report.json",
        PARENT_REG / "phase_b_final_audit.json",
        PARENT_REG / "phase_b_launcher_receipt.json",
        FAILED_ATTEMPT / "success_receipt.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"unexpected post-preflight artifact: {forbidden}")

    STAGING_REG.mkdir(parents=False, exist_ok=False)
    STAGING_AUDIT_DIR.mkdir(parents=False, exist_ok=False)
    STAGING_EVIDENCE_DIR.mkdir(parents=False, exist_ok=False)
    atomic_bytes(STAGING_AUDIT, Path(parent_audit["path"]).read_bytes())
    atomic_bytes(STAGING_OUTER_LOG, outer_log.read_bytes())
    if sha(STAGING_AUDIT) != parent_audit["sha256"] or sha(STAGING_OUTER_LOG) != sha(outer_log):
        raise RuntimeError("staged evidence copy drift")

    payload = copy.deepcopy(parent)
    payload["registry_root"] = str(NEW_REG.resolve())
    payload["execution_closure"]["registered_batch0_audit"] = {
        "path": str(FINAL_AUDIT.resolve()),
        "sha256": parent_audit["sha256"],
    }
    payload["execution_closure"].update(
        {
            "v481_parent_resume_preregistration": {"path": str(parent_path), "sha256": sha(parent_path)},
            "v481_failed_resume_launcher": old_launcher,
            "v481_failed_resume_attempt_intent": attempt_intent,
            "v481_failed_resume_receipt": failure_receipt,
            "v481_failed_resume_restore_log": restore_log,
            "v481_failed_resume_outer_log": {
                "path": str(FINAL_OUTER_LOG.resolve()),
                "sha256": sha(outer_log),
                "source_path_at_registration": str(outer_log),
            },
            "v481_resume_r2_materializer": {"path": str(materializer), "sha256": sha(materializer)},
        }
    )
    payload["lineage"] = {
        "format": "strict-track2-v481-v480-phase-b-resume-r2-lineage-v1",
        "parent_resume_preregistration_sha256": sha(parent_path),
        "parent_resume_launcher_sha256": old_launcher["sha256"],
        "failed_preflight_attempt_intent_sha256": attempt_intent["sha256"],
        "failed_preflight_receipt_sha256": failure_receipt["sha256"],
        "failed_preflight_outer_log_sha256": sha(outer_log),
        "failure_scope": "preflight_float_derived_difference_exact_dict_comparison_only",
        "simulator_or_generator_invocations": 0,
        "batch1_artifacts_created": False,
        "no_retry_of_failed_resume_attempt": True,
    }
    payload["resume_registration"].update(
        {
            "format": "strict-track2-v481-v480-phase-b-resume-r2-registration-v1",
            "authorized": True,
            "next_batch_id": 1,
            "batch0_reuse_required": True,
            "batch0_rerun_authorized": False,
            "batch0_generator_invocations_allowed": 0,
            "registered_batch0_audit": {"path": str(FINAL_AUDIT.resolve()), "sha256": parent_audit["sha256"]},
            "failed_parent_resume_attempt_bound": True,
            "preflight_formula_change": {
                "scope": "derived_accounting_difference_comparison_only",
                "comparison": "math.isclose",
                "relative_tolerance": 0.0,
                "absolute_tolerance": 1e-12,
                "all_wall_inputs_caps_and_gates_unchanged": True,
            },
        }
    )
    payload["resume_r2_attempt_accounting"] = {
        "original_launcher_raw_seconds": ORIGINAL_LAUNCHER_RAW,
        "original_inherited_watchdog_seconds_ceil": ORIGINAL_LAUNCHER_CEIL,
        "failed_resume_attempt_seconds": FAILED_ATTEMPT_WALL,
        "failed_resume_receipt_cumulative_raw_seconds": FAILED_CUMULATIVE_RAW,
        "effective_launcher_watchdog_prior_wall_seconds_ceil": NEXT_WATCHDOG_CEIL,
        "remaining_launcher_raw_seconds": WHOLE_WALL - FAILED_CUMULATIVE_RAW,
        "remaining_launcher_watchdog_integer_seconds": int(WHOLE_WALL) - NEXT_WATCHDOG_CEIL,
        "dataset_prior_cumulative_attempt_wall_seconds": DATASET_PRIOR,
        "effective_launcher_minus_dataset_raw_seconds": FAILED_CUMULATIVE_RAW - DATASET_PRIOR,
        "effective_watchdog_minus_dataset_seconds": NEXT_WATCHDOG_CEIL - DATASET_PRIOR,
        "derived_float_validation": {"comparison": "math.isclose", "relative_tolerance": 0.0, "absolute_tolerance": 1e-12},
        "v480_reconciliation_excluded_from_collection_wall": True,
    }
    payload["guards"].update(
        {
            "v481_failed_resume_preflight_bound": True,
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

    atomic_json(STAGING_FORMAL, payload)
    for directory in (STAGING_AUDIT_DIR, STAGING_EVIDENCE_DIR, STAGING_REG):
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
    if sha(FINAL_AUDIT) != parent_audit["sha256"] or sha(FINAL_OUTER_LOG) != sha(outer_log):
        raise RuntimeError("promoted evidence drift")
    print(json.dumps({"path": str(FINAL_FORMAL), "sha256": sha(FINAL_FORMAL), "registered_batch0_audit_sha256": sha(FINAL_AUDIT), "copied_outer_log_sha256": sha(FINAL_OUTER_LOG)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
