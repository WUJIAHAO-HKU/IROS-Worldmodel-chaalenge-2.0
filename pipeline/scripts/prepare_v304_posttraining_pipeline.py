#!/usr/bin/env python3
"""Preregister the outcome-free v304 post-training evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    off = args.base / "artifacts/strict_track2_official_20260810"
    scripts = args.base / "pipeline/scripts"
    dev = off / "real_robotwin_eval/public_unseen_train_dev112_seed1403"
    dev_out = off / "real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
    variant = "v305_v304_v301_fullbudget_step10_seed1471"
    freeze_dir = off / "frozen_candidates/v307_v304_v301_fullbudget_step10_seed1471"
    final_out = (
        off
        / "real_robotwin_eval/frozen_v307_v304_v301_fullbudget_step10_seed1471_final128"
    )
    seed_bundle = off / "real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
    training_prereg = args.registry / "preregistration.json"
    public_preflight = args.run / "audit/v304_public112_static_preflight.json"
    final_preflight = args.run / "audit/v304_final128_static_preflight.json"
    numeric_clarification = args.run / "audit/v303_numeric_gate_clarification.json"
    for path in (
        training_prereg,
        public_preflight,
        final_preflight,
        numeric_clarification,
        dev / "manifest.json",
        dev / "batch_00.json",
        seed_bundle,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    training = json.loads(training_prereg.read_text())
    public_static = json.loads(public_preflight.read_text())
    final_static = json.loads(final_preflight.read_text())
    numeric = json.loads(numeric_clarification.read_text())
    seeds = json.loads(seed_bundle.read_text())
    assert training["guards"]["official_rule_compliant"] is True
    assert training["guards"]["participant_modifies_only_world_model"] is True
    assert public_static["passed"] is True and public_static["outcomes_read"] is False
    assert final_static["passed"] is True and final_static["outcomes_read"] is False
    assert numeric["passed"] is True and numeric["runtime_or_training_changed"] is False
    assert seeds["result_data_used"] is False
    assert seeds["seed_count"] == 128 and seeds["unique_seed_count"] == 128
    assert not (dev_out / variant).exists()
    assert not freeze_dir.exists()
    assert not final_out.exists()

    tool_names = [
        "prepare_v305_v304_batch00_gate.py",
        "launch_v305_v304_batch00_gate.sh",
        "watch_v304_to_v305_pipeline.sh",
        "audit_v228_public_right_gate.py",
        "audit_v276_v274_public112_gate.py",
        "launch_v306_v304_public112_freeze_final.sh",
        "watch_v305_to_v307_pipeline.sh",
        "freeze_v307_v304_candidate.py",
        "prepare_frozen_final128_prereg.py",
        "run_frozen_v307_v304_final128_once.sh",
        "run_strict_track2_dev_eval.sh",
        "run_strict_track2_final128_eval.sh",
        "summarize_frozen_final128.py",
        "verify_v304_posttraining_tools.py",
        "prepare_v304_posttraining_pipeline.py",
    ]
    tools = {}
    for name in tool_names:
        path = scripts / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        tools[name] = {"path": str(path), "sha256": sha256(path)}

    record = {
        "format": "strict-track2-v304-posttraining-pipeline-preregistration-v2",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "training_preregistration": {
            "path": str(training_prereg),
            "sha256": sha256(training_prereg),
        },
        "stage_sequence": [
            "complete all 10 preregistered RL updates",
            "pass full training metric and checkpoint audit",
            "evaluate public batch00 only",
            "if and only if batch00 passes, evaluate remaining public112 batches",
            "if and only if public112 passes, freeze exactly one candidate",
            "preregister and execute exactly one local final128 evaluation",
        ],
        "batch00_gate": {
            "count": 16,
            "left_count": 4,
            "right_count": 12,
            "thresholds": {
                "total_success": 10,
                "left_success": 3,
                "right_success": 6,
                "grasp_once": 14,
                "right_grasp": 10,
            },
        },
        "public112_gate": {
            "count": 112,
            "thresholds": {
                "total_success": 75,
                "left_success_rate": 0.60,
                "right_success_rate": 0.60,
                "grasp_at_least_success": True,
            },
        },
        "final128_gate": {
            "count": 128,
            "successes_min": 85,
            "one_unique_frozen_candidate": True,
            "selection_after_final": False,
        },
        "public_manifest": {
            "path": str(dev / "manifest.json"),
            "sha256": sha256(dev / "manifest.json"),
        },
        "public_batch00": {
            "path": str(dev / "batch_00.json"),
            "sha256": sha256(dev / "batch_00.json"),
        },
        "final_seed_bundle": {
            "path": str(seed_bundle),
            "sha256": sha256(seed_bundle),
            "result_data_used": False,
        },
        "static_evidence": {
            "public112_preflight": {
                "path": str(public_preflight),
                "sha256": sha256(public_preflight),
            },
            "final128_preflight": {
                "path": str(final_preflight),
                "sha256": sha256(final_preflight),
            },
            "v303_numeric_gate_clarification": {
                "path": str(numeric_clarification),
                "sha256": sha256(numeric_clarification),
            },
        },
        "tools": tools,
        "guards": {
            "hidden_or_final_outcomes_used_for_training_or_selection": False,
            "public_policy_outcomes_read_at_registration": False,
            "real_submission": False,
            "participant_component": "world-model RGB service only",
            "policy_mirroring": False,
            "sft_co_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
