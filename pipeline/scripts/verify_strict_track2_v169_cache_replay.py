#!/usr/bin/env python3
"""Replay packaged V16.9 against the cache admitted with the same router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_strict_track2_autoregressive_candidate import Windows
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite replay report: {args.output}")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dataset = Windows(args.windows, split["validation_episodes"])
    by_name = {path.name: index for index, path in enumerate(dataset.paths)}
    with np.load(args.cache, allow_pickle=False) as values:
        paths = values["path"].astype(str).tolist()
        expected = values["candidate"].copy()
        expected_route = values["route_right"].astype(bool)
        instructions = values["instruction"].astype(str).tolist()
    runtime = Track2V169ArmRoutedRuntime(args.release, args.library, args.device)
    total_values = different_values = absolute_error_sum = max_error = route_errors = 0
    for position, name in enumerate(paths):
        context, history, future, _, _, _ = dataset[by_name[name]]
        instruction = instructions[position]
        route = bool(runtime.router.predict(context, history, future, instruction))
        route_errors += int(route != bool(expected_route[position]))
        actual = runtime.predict(context, history, future, 0, instruction)
        difference = np.abs(actual.astype(np.int16) - expected[position].astype(np.int16))
        total_values += difference.size
        different_values += int(np.count_nonzero(difference))
        absolute_error_sum += int(difference.sum())
        max_error = max(max_error, int(difference.max()))
        if position == 0 or (position + 1) % 8 == 0 or position + 1 == len(paths):
            print(json.dumps({"replayed": position + 1, "total": len(paths), "different_values": different_values}), flush=True)
    report = {
        "format": "strict-track2-v169-runtime-cache-replay-v1",
        "passed": different_values == 0 and route_errors == 0,
        "samples": len(paths),
        "route_errors": route_errors,
        "compared_values": total_values,
        "different_values": different_values,
        "different_fraction": different_values / total_values,
        "uint8_mae": absolute_error_sum / total_values,
        "uint8_max_absolute_error": max_error,
        "release": str(args.release.resolve()),
        "cache": str(args.cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("packaged V16.9 runtime is not bit-exact to its selection cache")


if __name__ == "__main__":
    main()
