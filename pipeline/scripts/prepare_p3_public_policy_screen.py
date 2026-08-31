#!/usr/bin/env python3
"""Freeze the v203b public-policy screen before simulator outcomes are read."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-audit", required=True, type=Path)
    parser.add_argument("--training-preregistration", required=True, type=Path)
    parser.add_argument("--dev32-manifest", required=True, type=Path)
    parser.add_argument("--dev112-manifest", required=True, type=Path)
    parser.add_argument("--variant", default="v203b_step8_seed1403")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    training_audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    if training_audit.get("passed") is not True:
        raise RuntimeError("training audit did not pass")
    dev32 = json.loads(args.dev32_manifest.read_text(encoding="utf-8"))
    dev112 = json.loads(args.dev112_manifest.read_text(encoding="utf-8"))
    if dev32["selected_seeds"] != dev112["selected_seeds"][:32]:
        raise RuntimeError("stage-A seeds are not the prefix of the frozen 112 split")
    for name, manifest, expected in (("dev32", dev32, 32), ("dev112", dev112, 112)):
        if int(manifest["count"]) != expected:
            raise RuntimeError(f"{name} count mismatch")
        if manifest["intersection_with_world_model_corpus"]:
            raise RuntimeError(f"{name} intersects world-model corpus")
        if manifest["intersection_with_official_eval_seeds"]:
            raise RuntimeError(f"{name} intersects official eval seeds")

    payload = {
        "format": "strict-track2-public-policy-screen-preregistration-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "variant": args.variant,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "training_audit": str(args.training_audit),
            "training_audit_sha256": sha256(args.training_audit),
            "training_preregistration": str(args.training_preregistration),
            "training_preregistration_sha256": sha256(args.training_preregistration),
        },
        "baseline": "official unmodified Pi0.5 checkpoint",
        "simulator": "public local RoboTwin adjust_bottle; no contest endpoint",
        "stage_a": {
            "manifest": str(args.dev32_manifest),
            "manifest_sha256": sha256(args.dev32_manifest),
            "count": 32,
            "batches": ["00", "01"],
            "gate": {
                "candidate_successes_min": 19,
                "success_gain_over_baseline_min": 2,
                "left_success_rate_min": 0.50,
                "right_success_rate_min": 0.50,
                "grasp_completions_not_below_baseline": True,
            },
            "decision": "reject without evaluating the remaining 80 public seeds if any gate fails",
        },
        "stage_b": {
            "manifest": str(args.dev112_manifest),
            "manifest_sha256": sha256(args.dev112_manifest),
            "count": 112,
            "additional_batches_after_stage_a": ["02", "03", "04", "05", "06"],
            "gate": {
                "candidate_successes_min": 75,
                "success_gain_over_baseline_min": 3,
                "left_success_rate_min": 0.60,
                "right_success_rate_min": 0.60,
                "grasp_completions_at_least_successes": True,
            },
            "target_mapping": "75/112=66.96%, slightly above the final target 85/128=66.41%",
        },
        "freeze_rule": "freeze as the unique final candidate only if both stages pass",
        "prohibited": [
            "official evaluation seeds or outcomes for selection",
            "hidden development outcomes",
            "real contest submission",
            "running the reserved final 128 before unique-candidate freeze",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
