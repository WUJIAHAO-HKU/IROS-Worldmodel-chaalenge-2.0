#!/usr/bin/env python3
"""Freeze the training-only v327 continuation decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_scalar(events: list[Path], tag: str) -> float:
    values = []
    for path in events:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags().get("scalars", []):
            values.extend(accumulator.Scalars(tag))
    if not values:
        raise RuntimeError(f"missing scalar: {tag}")
    item = max(values, key=lambda value: (int(value.step), float(value.wall_time)))
    return float(item.value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--training-acceptance", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    acceptance = json.loads(args.training_acceptance.read_text())
    if prereg.get("format") != "strict-track2-v327-v169-v326-one-update-preregistration-v1":
        raise RuntimeError("wrong preregistration")
    events = sorted((args.run / "tensorboard").glob("events.out.tfevents.*"))
    metrics = {
        tag: latest_scalar(events, tag)
        for tag in (
            "env/episode_len",
            "env/num_trajectories",
            "env/return",
            "env/reward",
            "env/success_once",
            "train/actor/approx_kl",
            "train/actor/clip_fraction",
            "train/actor/grad_norm",
            "train/actor/kl_loss",
        )
    }
    trajectories = int(round(metrics["env/num_trajectories"]))
    successes = int(round(metrics["env/success_once"] * trajectories))
    threshold = float(
        prereg["selection_protocol"]["initial_rollout_success_rate_go_signal"]
    )
    baseline = float(
        prereg["selection_protocol"]["comparison_baseline_v318_step0_success_rate"]
    )
    training_valid = (
        acceptance.get("passed") is True
        and trajectories == 128
        and all(math.isfinite(value) for value in metrics.values())
        and args.checkpoint.stat().st_size == 8_529_316_588
        and zipfile.is_zipfile(args.checkpoint)
    )
    go = training_valid and metrics["env/success_once"] >= threshold
    report = {
        "format": "strict-track2-v327-training-go-no-go-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run),
        "metrics": metrics,
        "successes": successes,
        "trajectories": trajectories,
        "success_fraction": f"{successes}/{trajectories}",
        "go_threshold_rate": threshold,
        "go_threshold_count": int(math.ceil(threshold * trajectories)),
        "v318_step0_baseline_rate": baseline,
        "v318_step0_baseline_count": int(round(baseline * trajectories)),
        "improvement_vs_v318_step0_count": successes - int(round(baseline * trajectories)),
        "checks": {
            "training_acceptance_passed": acceptance.get("passed") is True,
            "trajectory_budget_exact_128": trajectories == 128,
            "all_metrics_finite": all(math.isfinite(value) for value in metrics.values()),
            "checkpoint_size_exact": args.checkpoint.stat().st_size == 8_529_316_588,
            "checkpoint_zip_integrity": zipfile.is_zipfile(args.checkpoint),
            "initial_rollout_success_rate_ge_0p25": metrics["env/success_once"] >= threshold,
        },
        "training_valid": training_valid,
        "continuation_authorized": go,
        "public_batch16_authorized": go,
        "decision": "advance_to_preregistered_public_batch16" if go else "reject_before_public_evaluation",
        "reason": None if go else "initial rollout success signal is below the preregistered 32/128 gate",
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "training_acceptance": sha256(args.training_acceptance),
            "checkpoint": sha256(args.checkpoint),
            "event_files": {path.name: sha256(path) for path in events},
        },
        "guards": {
            "training_metrics_only": True,
            "public_evaluation_read_for_decision": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
