#!/usr/bin/env python3
"""Stop a policy candidate when dev22 batch01 cannot rescue batch00."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from summarize_strict_track2_real_eval import parse_log


def ordered_seeds(dev_root: Path, batch: str) -> list[int]:
    data = json.loads((dev_root / f"batch_{batch}.json").read_text(encoding="utf-8"))
    seeds = torch.as_tensor(data["adjust_bottle"]["success_seeds"], dtype=torch.long)
    order = torch.randperm(len(seeds), generator=torch.Generator().manual_seed(0))
    return [int(value) for value in seeds[order].tolist()]


def decode(output_root: Path, dev_root: Path, variant: str, batch: str, count: int) -> list[dict]:
    path = output_root / variant / f"batch_{batch}" / "launcher.log"
    metrics = parse_log(path)
    if metrics is None or int(metrics["num_trajectories"]) != count:
        raise RuntimeError(f"missing completed metrics in {path}")
    masks = {
        key: round(float(metrics[key]) * count)
        for key in ("success_bitmask", "grasp_bitmask", "arm_right_bitmask")
    }
    return [
        {
            "seed": seed,
            "arm": "right" if (masks["arm_right_bitmask"] >> index) & 1 else "left",
            "success": bool((masks["success_bitmask"] >> index) & 1),
            "grasp": bool((masks["grasp_bitmask"] >> index) & 1),
        }
        for index, seed in enumerate(ordered_seeds(dev_root, batch))
    ]


def counts(rows: list[dict], arm: str | None = None) -> dict[str, int]:
    selected = rows if arm is None else [row for row in rows if row["arm"] == arm]
    return {
        "count": len(selected),
        "successes": sum(row["success"] for row in selected),
        "grasps": sum(row["grasp"] for row in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.dev_root / "manifest.json").read_text(encoding="utf-8"))
    batch_counts = {f'{int(row["batch"]):02d}': int(row["count"]) for row in manifest["batches"]}
    if set(batch_counts) != {"00", "01"}:
        raise RuntimeError(f"unexpected dev22 batches: {batch_counts}")
    baseline = []
    for batch, count in batch_counts.items():
        baseline.extend(decode(args.output_root, args.dev_root, "baseline", batch, count))
    candidate = decode(args.output_root, args.dev_root, args.candidate, "00", batch_counts["00"])
    remaining = decode(args.output_root, args.dev_root, "baseline", "01", batch_counts["01"])

    scopes = {}
    checks = {}
    for arm in (None, "left", "right"):
        name = "all" if arm is None else arm
        base = counts(baseline, arm)
        current = counts(candidate, arm)
        possible = counts(remaining, arm)
        max_successes = current["successes"] + possible["count"]
        max_grasps = current["grasps"] + possible["count"]
        required_successes = base["successes"] + (1 if arm is None else 0)
        scopes[name] = {
            "baseline": base,
            "candidate_batch00": current,
            "remaining_count": possible["count"],
            "maximum_possible_successes": max_successes,
            "maximum_possible_grasps": max_grasps,
            "required_successes": required_successes,
            "required_grasps": base["grasps"],
        }
        checks[f"{name}_success_still_possible"] = max_successes >= required_successes
        checks[f"{name}_grasp_nonregression_still_possible"] = max_grasps >= base["grasps"]

    passed = all(checks.values())
    report = {
        "format": "strict-track2-dev22-batch00-mathematical-screen-v1",
        "candidate": args.candidate,
        "passed": passed,
        "scopes": scopes,
        "checks": checks,
        "development32_consumed": False,
        "acceptance128_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
