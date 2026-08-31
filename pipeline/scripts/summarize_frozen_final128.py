#!/usr/bin/env python3
"""Summarize one frozen candidate's local official-seed evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from summarize_strict_track2_real_eval import parse_log, wilson


EXPECTED = {"00": 8, "01": 16, "02": 16, "03": 16, "04": 16,
            "05": 16, "06": 16, "07": 16, "08": 8}


def shuffled_seeds(seed_root: Path, batch: str) -> list[int]:
    path = (seed_root / "seed_shards" / "shard_00.json" if batch == "00"
            else seed_root / "seed_batches16" / f"batch_{batch}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = torch.as_tensor(data["adjust_bottle"]["success_seeds"], dtype=torch.long)
    order = torch.randperm(len(seeds), generator=torch.Generator().manual_seed(0))
    return [int(seed) for seed in seeds[order].tolist()]


def scope(rows: list[dict], arm: str | None = None) -> dict:
    selected = rows if arm is None else [row for row in rows if row["arm"] == arm]
    successes = sum(row["success"] for row in selected)
    grasps = sum(row["grasp"] for row in selected)
    count = len(selected)
    return {
        "count": count,
        "successes": successes,
        "success_rate": successes / count,
        "success_rate_wilson_95": wilson(successes, count),
        "grasp_completions": grasps,
        "grasp_completion_rate": grasps / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--seed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("authorized") is not True or prereg.get("contest_submission") is not False:
        raise RuntimeError("invalid final-128 preregistration")
    variant = prereg["variant"]
    rows: list[dict] = []
    batches: list[dict] = []
    for batch, expected in EXPECTED.items():
        log = args.output_root / variant / f"batch16_{batch}" / "launcher.log"
        metrics = parse_log(log)
        if metrics is None or metrics["num_trajectories"] != expected:
            raise RuntimeError(f"missing complete final batch {batch}")
        packed = {
            key: round(metrics[key] * expected)
            for key in ("success_bitmask", "grasp_bitmask", "arm_right_bitmask")
        }
        decoded = []
        for index, seed in enumerate(shuffled_seeds(args.seed_root, batch)):
            row = {
                "seed": seed,
                "arm": "right" if (packed["arm_right_bitmask"] >> index) & 1 else "left",
                "success": bool((packed["success_bitmask"] >> index) & 1),
                "grasp": bool((packed["grasp_bitmask"] >> index) & 1),
            }
            if row["success"] and not row["grasp"]:
                raise RuntimeError(f"success without grasp for seed {seed}")
            decoded.append(row)
            rows.append(row)
        batches.append({
            "batch": batch,
            "count": expected,
            "successes": sum(row["success"] for row in decoded),
            "grasps": sum(row["grasp"] for row in decoded),
            "log": str(log),
        })
    overall = scope(rows)
    target = int(prereg["target"]["successes_min"])
    report = {
        "format": "strict-track2-frozen-local-final128-result-v1",
        "variant": variant,
        "checkpoint_sha256": prereg["checkpoint_sha256"],
        "official_submission": False,
        "evaluation_count": len(rows),
        "overall": overall,
        "left": scope(rows, "left"),
        "right": scope(rows, "right"),
        "batches": batches,
        "target_successes": target,
        "target_reached": overall["successes"] >= target,
        "selection_after_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
