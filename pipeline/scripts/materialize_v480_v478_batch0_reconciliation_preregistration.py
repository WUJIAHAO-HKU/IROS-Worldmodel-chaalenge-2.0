#!/usr/bin/env python3
"""Freeze the one-shot, read-only v478 batch-0 audit reconciliation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OLD_REG = J / "v478_temporal200_seed1622_20260824"
ATTEMPT = OLD_REG / "phase_b_launcher_attempts/20260823T234034-277910"
DATASET = Path("/root/v478_temporal8_dataset_seed1622_20260824")
NEW_REG = J / "v480_v478_phase_b_batch0_reconciliation_20260824"
OUTPUT = NEW_REG / "preregistration.json"
PIPELINE = ROOT / "pipeline/scripts"
V479 = J / "v479_v478_endpoint_oracle12_reconciliation_20260824"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bound(path: Path, expected: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file() or sha(resolved) != expected:
        raise RuntimeError(f"immutable evidence drift: {resolved}")
    return {"path": str(resolved), "sha256": expected}


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if NEW_REG.exists():
        raise FileExistsError(NEW_REG)
    evidence = {
        "reconciliation_contract": bound(
            PIPELINE / "v480_v478_batch0_immutable_reconciliation_contract.json",
            "2a492ddd0b974a74c5cbd8f1d11b093905c74f931da76abd745d94fea6aec827",
        ),
        "materializer": bound(Path(__file__), sha(Path(__file__))),
        "reconciler": bound(
            PIPELINE / "audit_v480_v478_phase_b_batch0_reconciliation.py",
            "5afc67521bc35f2de8636a972041319b47874f80565f029d75f9577c9a2edd06",
        ),
        "phase_b_preregistration": bound(
            OLD_REG / "phase_b_preregistration.json",
            "5d6db6e5aca4502b79347620a7732e4471f63b77780ad7228bcd53c6cb753ad9",
        ),
        "phase_b_contract": bound(
            PIPELINE / "v478_temporal200_phase_b_frozen_contract_r4.json",
            "a478b6f8e24d4a2312fcd4f62d294ad681b35ea4c19c1162f7dae7f9748047d1",
        ),
        "selection": bound(
            OLD_REG / "selection.json",
            "f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398",
        ),
        "selection_receipt": bound(
            OLD_REG / "selection_receipt.json",
            "c14b604b65ffcc13cd23f66a67260f92d3f495f85d7ecf3c823586e3242926b3",
        ),
        "endpoint_oracle_union": bound(
            OLD_REG / "phase_b_oracle_union.json",
            "0d3b1328709ba1b09bb29b1423596e5671fdc37c49215ea4b4f168b9b5fd53cc",
        ),
        "v479_preregistration": bound(
            V479 / "preregistration_r3.json",
            "07b6d5974c5891068160ec9ffc69a14717cb110de0009393ad5f051473cb9ae1",
        ),
        "v479_reconciliation": bound(
            V479 / "reconciliation_receipt.json",
            "12ce84ccf3cbf43720f0be0a1e857fcd2c160628a2ab169b006ef8704995b9e3",
        ),
        "v479_launcher": bound(
            V479 / "launcher_receipt.json",
            "ed0a4e1efeca71ac28ed300a2e79f983bb0da2a3bbbe75060da0075b75897159",
        ),
        "v479_recomputed_audit": bound(
            V479 / "recomputed_v478_audit_receipt.json",
            "e75cdcbf6be9f8289d677babf4d6453327ad260d53e77245c6a39424a8485dad",
        ),
        "generator": bound(
            PIPELINE / "generate_v478_temporal200_sharded.py",
            "a20f5e9d702fa6febe946d9f713f4713ef035705b35119c00ca328046da70a58",
        ),
        "legacy_auditor": bound(
            PIPELINE / "audit_v478_temporal200_sharded.py",
            "2e1adbe72e2d77d19a2bf6f4a3d0822940ca458055851c36e06b28fe62c6c748",
        ),
        "fixed_auditor": bound(
            PIPELINE / "audit_v480_v478_temporal200_sharded.py",
            "0557fcd60eb97f3577998315a4a78a4ee14d1e9f58a1383c5ba1266c1543b6c0",
        ),
        "collector": bound(
            PIPELINE / "collect_v477_temporal_paired.py",
            "be90f3dd1272706af6d9bc1f461543c9b399071139a4ea243bd962b4b01f187a",
        ),
        "legacy_launcher": bound(
            PIPELINE / "launch_v478_temporal200_phase_b.sh",
            "7a757d002d3ee4eaaff6eba339134f7e551260c805e258aa345f5c527e651997",
        ),
        "attempt_intent": bound(
            ATTEMPT / "attempt_intent.json",
            "dfd2fc285ba00df97950ae362a3754eabcf3816e8151cc75a23bd65609d97dd5",
        ),
        "launcher_failure": bound(
            ATTEMPT / "failure_receipt.json",
            "48c95faeda8992ebf041d677a1b3e231b4619269f5e86574c3f8a73473ac5452",
        ),
        "legacy_collect_log": bound(
            ATTEMPT / "collect_batch_000.log",
            "8deb88b051d19ab1070e1078f3e3dccff6ad45dce4d150827bf1b35a71c4f13d",
        ),
        "legacy_audit_log": bound(
            ATTEMPT / "audit_batch_000.log",
            "2e6177519ab32b6c0a898129239d8859ca555c04364591a4eb17ca99edbf1c01",
        ),
        "restore_v218_log": bound(
            ATTEMPT / "restore_v218_00.log",
            "bf36c8bfe0d2f0af1690b20840a80e622e67e3029daab4b4985a88fc2989acb1",
        ),
        "legacy_audit_tmp": bound(
            OLD_REG / "phase_b_batch_audits/batch_000_audit.json.tmp",
            "d85fcec41d978d991a86ea74b22d47b649fc02b012bad3c104ffde0301d47359",
        ),
        "batch_report": bound(
            DATASET / "batch_000/batch_report.json",
            "59f77e4f059732ea49bd922bf74551cd93d05248ebd121fc6741e645eb02f7dc",
        ),
    }
    required_absence = [
        OLD_REG / "phase_b_batch_audits/batch_000_audit.json",
        DATASET / "generation_report.json",
        OLD_REG / "phase_b_final_audit.json",
        ATTEMPT / "success_receipt.json",
        OLD_REG / "phase_b_launcher_receipt.json",
        DATASET / "batch_001",
        DATASET / "batch_001.partial",
    ]
    if any(path.exists() for path in required_absence):
        raise RuntimeError("legacy formal success or batch1 artifact unexpectedly exists")
    NEW_REG.mkdir(parents=True, exist_ok=False)
    payload = {
        "format": "strict-track2-v480-v478-phase-b-batch0-reconciliation-preregistration-v1",
        "status": "preregistered_immutable_batch0_audit_reconciliation_authorized",
        "registry_root": str(NEW_REG.resolve()),
        "batch0_root": str((DATASET / "batch_000").resolve()),
        "execution_closure": {
            "reconciler": evidence["reconciler"],
            "fixed_auditor": evidence["fixed_auditor"],
            "legacy_auditor": evidence["legacy_auditor"],
            "reconciliation_contract": evidence["reconciliation_contract"],
            "materializer": evidence["materializer"],
        },
        "immutable_evidence": evidence,
        "required_absence": [str(path.resolve()) for path in required_absence],
        "batch0_tree": {
            "file_count": 41,
            "logical_regular_file_bytes": 201004510,
            "sorted_sha256_double_space_relative_posix_lines_sha256": "b9a44ba8b63c2cc721e6bd94d03d38efff61f18fa32fe437904faf88e3e2d031",
            "canonical_compact_json_relative_sha256_pairs_sha256": "0ff88bdbd6196d33d94796d7733fdc81bd2102a55a8c53d611c1a9454414d1b5",
            "batch_report_sha256": "59f77e4f059732ea49bd922bf74551cd93d05248ebd121fc6741e645eb02f7dc",
        },
        "outputs": {
            "recomputed_batch0_audit": str((NEW_REG / "recomputed_batch_000_audit.json").resolve()),
            "reconciliation_receipt": str((NEW_REG / "reconciliation_receipt.json").resolve()),
        },
        "authorization_before_reconciliation": {
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
        },
        "guards": {
            "read_only_parent_dataset_and_attempt": True,
            "outputs_only_in_new_registry": True,
            "simulator_or_collector_invocations": 0,
            "batch0_generator_invocations": 0,
            "task_reward_success_done_outcome_consumed": False,
            "hidden_final_submission_consumed": False,
            "training_authorized": False,
            "s1_authorized": False,
            "zero_update_authorized": False,
            "policy_updates": 0,
            "rl_authorized": False,
        },
    }
    atomic(OUTPUT, payload)
    print(json.dumps({"path": str(OUTPUT), "sha256": sha(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
