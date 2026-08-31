#!/usr/bin/env python3
"""Freeze the sole v274 retry1 candidate after both public gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--public-preregistration", required=True, type=Path)
    parser.add_argument("--batch00-report", required=True, type=Path)
    parser.add_argument("--public112-report", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--parent-runtime", required=True, type=Path)
    parser.add_argument("--final-runner", required=True, type=Path)
    parser.add_argument("--final-summarizer", required=True, type=Path)
    parser.add_argument("--final-preregistration-tool", required=True, type=Path)
    parser.add_argument("--final-wrapper", required=True, type=Path)
    parser.add_argument("--final-static-preflight", required=True, type=Path)
    parser.add_argument("--public-static-preflight", required=True, type=Path)
    parser.add_argument("--state-alignment-audit", type=Path)
    parser.add_argument("--cpu-limit-amendment", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.checkpoint, args.training_report, args.public_preregistration,
        args.batch00_report, args.public112_report, args.parent_manifest,
        args.parent_runtime, args.final_runner, args.final_summarizer,
        args.final_preregistration_tool, args.final_wrapper,
        args.final_static_preflight,
        args.public_static_preflight,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    prereg = json.loads(args.public_preregistration.read_text(encoding="utf-8"))
    first = json.loads(args.batch00_report.read_text(encoding="utf-8"))
    public112 = json.loads(args.public112_report.read_text(encoding="utf-8"))
    final_preflight = json.loads(args.final_static_preflight.read_text(encoding="utf-8"))
    public_preflight = json.loads(args.public_static_preflight.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(args.checkpoint)
    if training.get("passed") is not True or training.get("checkpoint") != str(args.checkpoint):
        raise RuntimeError("training acceptance does not bind the checkpoint")
    if prereg.get("variant") != args.variant or prereg.get("checkpoint") != str(args.checkpoint):
        raise RuntimeError("public preregistration candidate mismatch")
    if prereg.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("public preregistration checkpoint hash mismatch")
    if first.get("accepted") is not True or public112.get("passed") is not True:
        raise RuntimeError("public gates did not both pass")
    if public112.get("variant") != args.variant or public112.get("count") != 112:
        raise RuntimeError("public112 candidate or count mismatch")
    if public112["candidate"]["all"]["successes"] < 75:
        raise RuntimeError("public112 absolute target not met")
    if final_preflight.get("passed") is not True or final_preflight.get("outcomes_read") is not False:
        raise RuntimeError("local final128 static preflight did not pass cleanly")
    if public_preflight.get("passed") is not True or public_preflight.get("outcomes_read") is not False:
        raise RuntimeError("public112 static preflight did not pass cleanly")
    if args.state_alignment_audit is not None:
        if not args.state_alignment_audit.is_file():
            raise FileNotFoundError(args.state_alignment_audit)
        state_alignment = json.loads(
            args.state_alignment_audit.read_text(encoding="utf-8")
        )
        if (
            state_alignment.get("passed") is not True
            or state_alignment.get("policy_outcomes_read") is not False
            or state_alignment.get("official_submission") is not False
        ):
            raise RuntimeError("HTTP state-alignment audit did not pass cleanly")
    if args.cpu_limit_amendment is not None:
        if not args.cpu_limit_amendment.is_file():
            raise FileNotFoundError(args.cpu_limit_amendment)
        cpu_amendment = json.loads(
            args.cpu_limit_amendment.read_text(encoding="utf-8")
        )
        if (
            cpu_amendment.get("passed") is not True
            or cpu_amendment.get("policy_or_data_changed") is not False
            or cpu_amendment.get("policy_outcomes_read") is not False
            or cpu_amendment.get("official_submission") is not False
        ):
            raise RuntimeError("CPU-limit restart amendment did not pass cleanly")

    evidence_paths = {
        "training_report": args.training_report,
        "public_preregistration": args.public_preregistration,
        "public_batch00_report": args.batch00_report,
        "public112_report": args.public112_report,
        "parent_world_model_manifest": args.parent_manifest,
        "parent_world_model_runtime": args.parent_runtime,
        "local_final128_runner": args.final_runner,
        "local_final128_summarizer": args.final_summarizer,
        "local_final128_preregistration_tool": args.final_preregistration_tool,
        "local_final128_one_candidate_wrapper": args.final_wrapper,
        "local_final128_static_preflight": args.final_static_preflight,
        "public112_static_preflight": args.public_static_preflight,
        "candidate_freeze_tool": Path(__file__).resolve(),
    }
    if args.state_alignment_audit is not None:
        evidence_paths["http_state_alignment_audit"] = args.state_alignment_audit
    if args.cpu_limit_amendment is not None:
        evidence_paths["cpu_limit_restart_amendment"] = args.cpu_limit_amendment
    payload = {
        "format": "strict-track2-v277-unique-final-candidate-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in evidence_paths.items()
        },
        "public_batch00_successes": first["counts"]["total_success"],
        "public112_successes": public112["candidate"]["all"]["successes"],
        "unique_final_candidate": True,
        "official_submission": False,
        "local_final128_runs_for_this_protocol_before_freeze": 0,
        "selection_inputs": "training diagnostics and public local train-seed outcomes only",
        "post_final_selection_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
