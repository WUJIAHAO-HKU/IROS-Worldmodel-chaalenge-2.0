#!/usr/bin/env python3
"""Build auditable per-output-frame blends from frozen baseline/candidate caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_schedule(value: str) -> tuple[float, ...]:
    schedule = tuple(float(item) for item in value.split(","))
    if len(schedule) != 8 or any(not 0.0 <= item <= 1.0 for item in schedule):
        raise argparse.ArgumentTypeError("schedule must contain eight values in [0, 1]")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cache", required=True, type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "A0,...,A7"),
        default=[],
    )
    parser.add_argument(
        "--arm-candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "LEFT_A0,...,A7", "RIGHT_A0,...,A7"),
        default=[],
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    schedules = {name: (parse_schedule(value), parse_schedule(value)) for name, value in args.candidate}
    schedules.update(
        (name, (parse_schedule(left), parse_schedule(right)))
        for name, left, right in args.arm_candidate
    )
    if not schedules:
        raise ValueError("at least one candidate is required")
    if len(schedules) != len(args.candidate) + len(args.arm_candidate):
        raise ValueError("candidate names must be unique")

    source = args.full_cache.resolve()
    with np.load(source, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    if args.baseline_cache:
        with np.load(args.baseline_cache.resolve(), allow_pickle=False) as values:
            for name in ("context_last", "target", "baseline", "arm_right", "capture_success", "path", "synthetic_seed", "start"):
                arrays[name] = values[name].copy()
    baseline = arrays["baseline"].astype(np.float32)
    full = arrays["candidate"].astype(np.float32)
    records = []
    right = arrays["arm_right"].astype(bool)
    for name, (left_schedule, right_schedule) in schedules.items():
        per_window = np.where(
            right[:, None],
            np.asarray(right_schedule, dtype=np.float32)[None, :],
            np.asarray(left_schedule, dtype=np.float32)[None, :],
        )
        alpha = per_window[:, :, None, None, None]
        candidate = np.clip(np.rint(baseline + alpha * (full - baseline)), 0, 255).astype(np.uint8)
        destination = output / f"{name}.npz"
        payload = dict(arrays)
        payload["candidate"] = candidate
        np.savez(destination, **payload)
        records.append(
            {
                "name": name,
                "left_schedule": left_schedule,
                "right_schedule": right_schedule,
                "cache": str(destination),
                "cache_sha256": sha256(destination),
            }
        )

    manifest = {
        "format": "strict-track2-temporal-output-blend-sweep-v1",
        "full_cache": str(source),
        "full_cache_sha256": sha256(source),
        "baseline_cache": str(args.baseline_cache.resolve()) if args.baseline_cache else str(source),
        "baseline_cache_sha256": sha256(args.baseline_cache.resolve()) if args.baseline_cache else sha256(source),
        "operation": "round_uint8(baseline + alpha[t] * (frozen_candidate - baseline))",
        "policy_action_selection": False,
        "mpc": False,
        "candidates": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
