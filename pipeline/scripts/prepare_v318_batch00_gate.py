#!/usr/bin/env python3
"""Preregister the first public-only gate for the v318/v317 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--training-preregistration", required=True, type=Path)
    parser.add_argument("--world-release", required=True, type=Path)
    parser.add_argument("--static-preflight", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch00", required=True, type=Path)
    parser.add_argument("--prepare-script", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    training = json.loads(args.training_report.read_text())
    training_prereg = json.loads(args.training_preregistration.read_text())
    release = json.loads(args.world_release.read_text())
    preflight = json.loads(args.static_preflight.read_text())
    assert training["passed"] is True
    assert training["checkpoint"] == str(args.checkpoint)
    assert args.checkpoint.stat().st_size > 8_000_000_000
    assert training_prereg["guards"]["official_rule_compliant"] is True
    assert training_prereg["guards"]["participant_modifies_only_world_model"] is True
    assert training_prereg["frozen_training"]["sft_co_training"] is False
    assert training_prereg["frozen_training"]["policy_mirroring"] is False
    assert training_prereg["world_model"]["model_version"] == (
        "track2-v317-batched-sparse-failure-terminal-v315"
    )
    assert preflight["passed"] is True
    assert preflight["outcomes_read"] is False
    assert preflight["official_submission"] is False

    thresholds = {
        "right_success": 6,
        "left_success": 3,
        "total_success": 10,
        "grasp_once": 14,
        "right_grasp": 10,
    }
    record = {
        "format": "strict-track2-v305-v304-public-screen-preregistration-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "purpose": (
            "first public-only screen after fresh official Pi0.5 RL training "
            "against the preregistered v317 world model"
        ),
        "selection_uses_policy_outcomes": False,
        "batch00_gate": {
            "batch": 0,
            "episode_composition": {"left": 4, "right": 12, "total": 16},
            "thresholds": thresholds,
            "on_reject": (
                "do not access batch01, later public batches, or reserved "
                "final128; revise only from permitted training/public data"
            ),
        },
        "thresholds": thresholds,
        "public112_gate": {
            "count": 112,
            "batches": ["00", "01", "02", "03", "04", "05", "06"],
            "additional_batches_after_batch00": [
                "01", "02", "03", "04", "05", "06"
            ],
            "thresholds": {
                "total_success": 75,
                "left_success_rate": 0.60,
                "right_success_rate": 0.60,
                "grasp_at_least_success": True,
            },
            "target_mapping": "75/112=66.96%, above 85/128=66.41%",
        },
        "freeze_rule": (
            "freeze exactly this candidate only after both public gates pass, "
            "then allow one local final128 evaluation"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "training_report": str(args.training_report),
        "training_report_sha256": sha256(args.training_report),
        "training_preregistration": str(args.training_preregistration),
        "training_preregistration_sha256": sha256(args.training_preregistration),
        "world_release": str(args.world_release),
        "world_release_sha256": sha256(args.world_release),
        "world_release_format": release.get("format"),
        "static_preflight": str(args.static_preflight),
        "static_preflight_sha256": sha256(args.static_preflight),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "batch00": str(args.batch00),
        "batch00_sha256": sha256(args.batch00),
        "prepare_script": str(args.prepare_script),
        "prepare_script_sha256": sha256(args.prepare_script),
        "launcher": str(args.launcher),
        "launcher_sha256": sha256(args.launcher),
        "real_submission": False,
        "reserved_final128_used": False,
        "batch01_or_later_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
