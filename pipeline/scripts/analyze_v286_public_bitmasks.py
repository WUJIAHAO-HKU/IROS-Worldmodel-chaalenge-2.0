#!/usr/bin/env python3
"""Decode comparable public batch00 bitmasks without running the simulator."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


METRIC = re.compile(r"'eval/(?P<name>[^']+)': array\((?P<value>[-+0-9.eE]+)")


def parse_variant(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("variant must be NAME=/absolute/path/eval.log")
    return name, Path(path)


def read_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    metrics = {match.group("name"): float(match.group("value")) for match in METRIC.finditer(text)}
    required = {"success_bitmask", "grasp_bitmask", "arm_right_bitmask"}
    missing = required - metrics.keys()
    if missing:
        raise RuntimeError(f"{path}: missing metrics {sorted(missing)}")
    trajectory_match = re.search(r"'eval/num_trajectories':\s*(\d+)", text)
    if trajectory_match is None or int(trajectory_match.group(1)) != 16:
        raise RuntimeError(f"{path}: expected 16 trajectories")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, type=parse_variant)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    seeds = torch.tensor([168, 169, 170, 171, 172, 173, 175, 176, 177, 179, 181, 182, 183, 184, 185, 186])
    order = seeds[torch.randperm(len(seeds), generator=torch.Generator().manual_seed(0))].tolist()
    parsed = {name: read_metrics(path) for name, path in args.variant}
    arm_masks = {round(metrics["arm_right_bitmask"] * 16) for metrics in parsed.values()}
    if len(arm_masks) != 1:
        raise RuntimeError(f"arm assignment masks differ: {sorted(arm_masks)}")
    arm_mask = next(iter(arm_masks))

    rows = []
    for index, seed in enumerate(order):
        row = {
            "rollout_index": index,
            "seed": seed,
            "arm": "right" if (arm_mask >> index) & 1 else "left",
            "variants": {},
        }
        for name, metrics in parsed.items():
            success_mask = round(metrics["success_bitmask"] * 16)
            grasp_mask = round(metrics["grasp_bitmask"] * 16)
            row["variants"][name] = {
                "success": bool((success_mask >> index) & 1),
                "grasp": bool((grasp_mask >> index) & 1),
            }
        rows.append(row)

    summaries = {}
    for name in parsed:
        success = [row for row in rows if row["variants"][name]["success"]]
        grasp = [row for row in rows if row["variants"][name]["grasp"]]
        summaries[name] = {
            "success_seeds": [row["seed"] for row in success],
            "right_success_seeds": [row["seed"] for row in success if row["arm"] == "right"],
            "left_success_seeds": [row["seed"] for row in success if row["arm"] == "left"],
            "grasp_fail_seeds": [row["seed"] for row in rows if row not in grasp],
        }

    report = {
        "format": "strict-track2-v286-public-batch00-bitmask-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_order": order,
        "summaries": summaries,
        "rows": rows,
        "rules": {
            "simulator_rerun": False,
            "existing_public_batch00_logs_only": True,
            "reserved_final128_access": False,
            "real_competition_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
