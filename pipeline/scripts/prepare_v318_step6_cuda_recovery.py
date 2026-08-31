#!/usr/bin/env python3
"""Register a fail-closed operational recovery from v318 global_step_6."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--step6-gate", required=True, type=Path)
    parser.add_argument("--failed-log", required=True, type=Path)
    parser.add_argument("--service-log", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--optimizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.preregistration,
        args.step6_gate,
        args.failed_log,
        args.service_log,
        args.checkpoint,
        args.optimizer,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    prereg = json.loads(args.preregistration.read_text())
    gate = json.loads(args.step6_gate.read_text())
    failed_text = args.failed_log.read_text(errors="replace")
    service_text = args.service_log.read_text(errors="replace")

    guards = prereg.get("guards", {})
    frozen = prereg.get("frozen_training", {})
    if not (
        guards.get("official_rule_compliant") is True
        and guards.get("participant_modifies_only_world_model") is True
        and guards.get("hidden_or_final_data_used_for_training_or_selection") is False
        and guards.get("policy_action_injection") is False
        and frozen.get("max_steps") == 10
        and frozen.get("actor_seed") == 1471
        and frozen.get("env_seed") == 0
        and frozen.get("actor_lr") == 2e-5
        and frozen.get("kl_beta") == 0.01
        and frozen.get("sft_co_training") is False
        and frozen.get("policy_mirroring") is False
    ):
        raise RuntimeError("frozen official-training guards changed")
    if gate.get("passed") is not True or not all(gate.get("checks", {}).values()):
        raise RuntimeError("global_step6 continuation gate is not clean")
    if not zipfile.is_zipfile(args.checkpoint) or not zipfile.is_zipfile(args.optimizer):
        raise RuntimeError("global_step6 recovery files are not valid ZIP checkpoints")
    if "HTTP Error 500" not in failed_text or "CUDA error: unknown error" not in service_text:
        raise RuntimeError("failed attempt does not prove the registered CUDA failure")
    if (args.run / "audit/v318_training_prefix_step6.json").exists():
        raise RuntimeError("step6 metrics already exist; partial rollout cannot be discarded")
    for forbidden in (
        args.run / "audit/V318_TRAINING_ACCEPTED",
        args.run / "audit/p3_training_acceptance.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"training already accepted: {forbidden}")

    payload = {
        "format": "strict-track2-v318-step6-cuda-reset-recovery-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "v317 CUDA context returned unknown error during an incomplete step6 "
            "rollout; fresh processes could not initialize CUDA, requiring an "
            "instance reset"
        ),
        "resume_from": "global_step_6 model plus optimizer",
        "discarded_work": {
            "global_step": 6,
            "partial_rollout_completed": False,
            "policy_update_applied": False,
            "metrics_logged_or_used_for_selection": False,
        },
        "checkpoint": {
            "model": str(args.checkpoint),
            "model_size": args.checkpoint.stat().st_size,
            "model_sha256": sha256(args.checkpoint),
            "optimizer": str(args.optimizer),
            "optimizer_size": args.optimizer.stat().st_size,
            "optimizer_sha256": sha256(args.optimizer),
            "zip_integrity": True,
        },
        "failed_attempt_log": {
            "path": str(args.failed_log),
            "sha256": sha256(args.failed_log),
        },
        "failed_service_log": {
            "path": str(args.service_log),
            "sha256": sha256(args.service_log),
        },
        "invariants": {
            "algorithm_changed": False,
            "world_model_changed": False,
            "same_run_directory": True,
            "same_model_and_optimizer_state": True,
            "same_actor_seed": 1471,
            "same_env_seed": 0,
            "same_horizon": 200,
            "same_rollouts_per_update": 128,
            "same_lr": 2e-5,
            "same_kl_beta": 0.01,
            "public112_or_final_outcomes_used": False,
            "real_submission": False,
        },
        "selection_inputs": (
            "checkpoint integrity, training-only step6 gate, and infrastructure "
            "error logs; no public112/final128/hidden/contest outcomes"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
