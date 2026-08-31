#!/usr/bin/env python3
"""Freeze the sole v304 candidate after both preregistered public gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--training-preregistration", required=True, type=Path)
    parser.add_argument("--posttraining-preregistration", required=True, type=Path)
    parser.add_argument("--public-preregistration", required=True, type=Path)
    parser.add_argument("--batch00-report", required=True, type=Path)
    parser.add_argument("--public112-report", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--parent-runtime", required=True, type=Path)
    parser.add_argument("--numeric-gate-clarification", required=True, type=Path)
    parser.add_argument("--public-runner", required=True, type=Path)
    parser.add_argument("--final-runner", required=True, type=Path)
    parser.add_argument("--final-summarizer", required=True, type=Path)
    parser.add_argument("--final-preregistration-tool", required=True, type=Path)
    parser.add_argument("--final-wrapper", required=True, type=Path)
    parser.add_argument("--public112-launcher", required=True, type=Path)
    parser.add_argument("--final-static-preflight", required=True, type=Path)
    parser.add_argument("--public-static-preflight", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = (
        args.checkpoint,
        args.training_report,
        args.training_preregistration,
        args.posttraining_preregistration,
        args.public_preregistration,
        args.batch00_report,
        args.public112_report,
        args.parent_manifest,
        args.parent_runtime,
        args.numeric_gate_clarification,
        args.public_runner,
        args.final_runner,
        args.final_summarizer,
        args.final_preregistration_tool,
        args.final_wrapper,
        args.public112_launcher,
        args.final_static_preflight,
        args.public_static_preflight,
    )
    for path in inputs:
        require_file(path)

    training = json.loads(args.training_report.read_text())
    training_prereg = json.loads(args.training_preregistration.read_text())
    posttraining_prereg = json.loads(args.posttraining_preregistration.read_text())
    public_prereg = json.loads(args.public_preregistration.read_text())
    batch00 = json.loads(args.batch00_report.read_text())
    public112 = json.loads(args.public112_report.read_text())
    parent = json.loads(args.parent_manifest.read_text())
    numeric = json.loads(args.numeric_gate_clarification.read_text())
    final_preflight = json.loads(args.final_static_preflight.read_text())
    public_preflight = json.loads(args.public_static_preflight.read_text())
    checkpoint_hash = sha256(args.checkpoint)

    if training.get("passed") is not True or training.get("checkpoint") != str(
        args.checkpoint
    ):
        raise RuntimeError("training acceptance does not bind the checkpoint")
    guards = training_prereg.get("guards", {})
    if not (
        guards.get("official_rule_compliant") is True
        and guards.get("participant_modifies_only_world_model") is True
        and guards.get("hidden_or_final_data_used_for_training_or_selection") is False
        and guards.get("policy_action_injection") is False
        and training_prereg["frozen_training"]["sft_co_training"] is False
        and training_prereg["frozen_training"]["policy_mirroring"] is False
    ):
        raise RuntimeError("training preregistration official guards failed")
    if (
        public_prereg.get("format")
        != "strict-track2-v305-v304-public-screen-preregistration-v1"
        or public_prereg.get("variant") != args.variant
        or public_prereg.get("checkpoint") != str(args.checkpoint)
        or public_prereg.get("checkpoint_sha256") != checkpoint_hash
        or public_prereg.get("real_submission") is not False
        or public_prereg.get("reserved_final128_used") is not False
    ):
        raise RuntimeError("public preregistration candidate mismatch")
    if not (
        posttraining_prereg.get("format")
        == "strict-track2-v304-posttraining-pipeline-preregistration-v2"
        and posttraining_prereg.get("variant") == args.variant
        and posttraining_prereg["final128_gate"]["successes_min"] == 85
        and posttraining_prereg["final128_gate"][
            "one_unique_frozen_candidate"
        ]
        is True
        and posttraining_prereg["final128_gate"]["selection_after_final"]
        is False
    ):
        raise RuntimeError("post-training pipeline preregistration mismatch")
    if (
        public_prereg.get("world_release") != str(args.parent_manifest)
        or public_prereg.get("world_release_sha256") != sha256(args.parent_manifest)
    ):
        raise RuntimeError("public preregistration world release mismatch")
    if parent.get("fixed_model", {}).get("model_version") != training_prereg[
        "world_model"
    ]["model_version"]:
        raise RuntimeError("training and release world-model versions differ")
    implementation = parent.get("implementation", {})
    if implementation.get(str(args.parent_runtime)) != sha256(args.parent_runtime):
        raise RuntimeError("v301 runtime is not bound by the world release")
    if not (
        numeric.get("passed") is True
        and numeric.get("runtime_or_training_changed") is False
        and numeric.get("selection_thresholds_changed_after_outcomes") is False
        and numeric.get("official_submission") is False
    ):
        raise RuntimeError("v303 numeric clarification did not pass cleanly")
    if not (
        batch00.get("accepted") is True
        and batch00.get("reserved_final128_used") is False
        and batch00.get("real_submission") is False
        and public112.get("passed") is True
        and public112.get("official_submission") is False
        and public112.get("reserved_final128_used") is False
    ):
        raise RuntimeError("public gates did not both pass cleanly")
    if public112.get("variant") != args.variant or public112.get("count") != 112:
        raise RuntimeError("public112 identity or count mismatch")
    if public112["candidate"]["all"]["successes"] < 75:
        raise RuntimeError("public112 absolute target not met")
    for preflight, label in (
        (final_preflight, "final128"),
        (public_preflight, "public112"),
    ):
        if not (
            preflight.get("passed") is True
            and preflight.get("outcomes_read") is False
            and preflight.get("official_submission") is False
        ):
            raise RuntimeError(f"{label} static preflight did not pass cleanly")
    if not (
        public_preflight.get("runner") == str(args.public_runner)
        and public_preflight.get("runner_sha256") == sha256(args.public_runner)
    ):
        raise RuntimeError("public runner changed after static preflight")
    if not (
        final_preflight.get("runner") == str(args.final_runner)
        and final_preflight.get("runner_sha256") == sha256(args.final_runner)
    ):
        raise RuntimeError("final runner changed after static preflight")

    evidence_paths = {
        "training_report": args.training_report,
        "training_preregistration": args.training_preregistration,
        "posttraining_pipeline_preregistration": args.posttraining_preregistration,
        "public_preregistration": args.public_preregistration,
        "public_batch00_report": args.batch00_report,
        "public112_report": args.public112_report,
        "parent_world_model_manifest": args.parent_manifest,
        "parent_world_model_runtime": args.parent_runtime,
        "v303_numeric_gate_clarification": args.numeric_gate_clarification,
        "public112_runner": args.public_runner,
        "local_final128_runner": args.final_runner,
        "local_final128_summarizer": args.final_summarizer,
        "local_final128_preregistration_tool": args.final_preregistration_tool,
        "local_final128_one_candidate_wrapper": args.final_wrapper,
        "public112_freeze_final_launcher": args.public112_launcher,
        "local_final128_static_preflight": args.final_static_preflight,
        "public112_static_preflight": args.public_static_preflight,
        "candidate_freeze_tool": Path(__file__).resolve(),
    }
    payload = {
        "format": "strict-track2-v307-v304-unique-final-candidate-freeze-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "world_model_version": parent["fixed_model"]["model_version"],
        "evidence": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in evidence_paths.items()
        },
        "public_batch00_successes": batch00["counts"]["total_success"],
        "public112_successes": public112["candidate"]["all"]["successes"],
        "unique_final_candidate": True,
        "official_submission": False,
        "local_final128_runs_for_this_protocol_before_freeze": 0,
        "selection_inputs": (
            "training diagnostics and public local train-seed outcomes only"
        ),
        "post_final_selection_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
