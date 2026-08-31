#!/usr/bin/env python3
"""Rank fixed-budget replicas on a preregistered disjoint development split."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from summarize_strict_track2_real_eval import exact_mcnemar, parse_log, rate_delta, wilson


def batch_counts(dev_root: Path) -> dict[str, int]:
    manifest = json.loads((dev_root / "manifest.json").read_text(encoding="utf-8"))
    counts = {
        f'{int(row["batch"]):02d}': int(row["count"])
        for row in manifest["batches"]
    }
    expected = manifest.get("count", manifest.get("development_count"))
    if expected is None or not counts or sum(counts.values()) != int(expected):
        raise RuntimeError("invalid preregistered development manifest")
    return counts


def ordered_seeds(dev_root: Path, batch: str) -> list[int]:
    data = json.loads((dev_root / f"batch_{batch}.json").read_text(encoding="utf-8"))
    seeds = torch.as_tensor(data["adjust_bottle"]["success_seeds"], dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    return [int(value) for value in seeds[torch.randperm(len(seeds), generator=generator)].tolist()]


def aggregate(output_root: Path, dev_root: Path, variant: str) -> dict:
    results = []
    batch_rows = []
    for batch, count in batch_counts(dev_root).items():
        log = output_root / variant / f"batch_{batch}" / "launcher.log"
        metrics = parse_log(log)
        if metrics is None or metrics["num_trajectories"] != count:
            raise RuntimeError(f"missing completed {variant} development batch {batch}")
        packed = {}
        for key in ("success_bitmask", "grasp_bitmask", "arm_right_bitmask"):
            if key not in metrics:
                raise RuntimeError(f"missing {key} in {log}")
            packed[key] = round(metrics[key] * count)
        decoded = []
        for index, seed in enumerate(ordered_seeds(dev_root, batch)):
            row = {
                "seed": seed,
                "arm": "right" if (packed["arm_right_bitmask"] >> index) & 1 else "left",
                "success": bool((packed["success_bitmask"] >> index) & 1),
                "grasp": bool((packed["grasp_bitmask"] >> index) & 1),
            }
            if row["success"] and not row["grasp"]:
                raise RuntimeError(f"success without grasp evidence for seed {seed}")
            decoded.append(row)
            results.append(row)
        expected_success = round(metrics["success_once"] * count)
        expected_grasp = round(metrics["grasp_once"] * count)
        if sum(row["success"] for row in decoded) != expected_success:
            raise RuntimeError(f"success bitmask mismatch for {variant} batch {batch}")
        if sum(row["grasp"] for row in decoded) != expected_grasp:
            raise RuntimeError(f"grasp bitmask mismatch for {variant} batch {batch}")
        batch_rows.append(
            {
                "batch": batch,
                "count": count,
                "successes": expected_success,
                "grasps": expected_grasp,
                "log_path": str(log),
            }
        )

    def scope(arm: str | None) -> dict:
        rows = results if arm is None else [row for row in results if row["arm"] == arm]
        successes = sum(row["success"] for row in rows)
        grasps = sum(row["grasp"] for row in rows)
        return {
            "count": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "success_rate_wilson_95": wilson(successes, len(rows)),
            "grasp_completions": grasps,
            "grasp_completion_rate": grasps / len(rows),
        }

    return {
        "all": scope(None),
        "left": scope("left"),
        "right": scope("right"),
        "batches": batch_rows,
        "seed_results": results,
    }


def comparison(baseline: dict, candidate: dict) -> dict:
    success_delta = {
        scope: rate_delta(baseline[scope]["success_rate"], candidate[scope]["success_rate"])
        for scope in ("all", "left", "right")
    }
    grasp_delta = {
        scope: rate_delta(
            baseline[scope]["grasp_completion_rate"],
            candidate[scope]["grasp_completion_rate"],
        )
        for scope in ("all", "left", "right")
    }
    paired = {
        scope: exact_mcnemar(
            baseline["seed_results"],
            candidate["seed_results"],
            arm=None if scope == "all" else scope,
        )
        for scope in ("all", "left", "right")
    }
    eligible = (
        success_delta["all"]["absolute_at_least_3pp"]
        and success_delta["all"]["relative_at_least_3_percent"]
        and success_delta["left"]["absolute_percentage_points"] >= 0.0
        and success_delta["right"]["absolute_percentage_points"] >= 0.0
        and all(grasp_delta[scope]["absolute_percentage_points"] >= 0.0 for scope in ("all", "left", "right"))
    )
    return {
        "eligible": eligible,
        "success_delta": success_delta,
        "grasp_delta": grasp_delta,
        "paired_success": paired,
    }


def rank_key(row: dict) -> tuple:
    metrics = row["metrics"]
    comparison_row = row["comparison"]
    paired = comparison_row["paired_success"]["all"]
    minimum_arm_delta = min(
        comparison_row["success_delta"][arm]["absolute_percentage_points"]
        for arm in ("left", "right")
    )
    match = re.search(r"(\d+)$", row["variant"])
    seed = int(match.group(1)) if match is not None else 2**31 - 1
    return (
        metrics["all"]["successes"],
        paired["baseline_fail_trained_success"] - paired["baseline_success_trained_fail"],
        minimum_arm_delta,
        metrics["all"]["grasp_completions"],
        -seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "--eval-root", required=True, type=Path)
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--candidate", "--candidate-variant", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = aggregate(args.output_root, args.dev_root, "baseline")
    candidates = []
    for variant in args.candidate:
        metrics = aggregate(args.output_root, args.dev_root, variant)
        candidates.append(
            {
                "variant": variant,
                "metrics": metrics,
                "comparison": comparison(baseline, metrics),
            }
        )
    eligible = sorted(
        (row for row in candidates if row["comparison"]["eligible"]),
        key=rank_key,
        reverse=True,
    )
    report = {
        "format": "strict-track2-dev-replica-ranking-v1",
        "seed_count": sum(batch_counts(args.dev_root).values()),
        "baseline": baseline,
        "candidates": candidates,
        "eligible_variants_ranked": [row["variant"] for row in eligible],
        "selected_variant": eligible[0]["variant"] if eligible else None,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
