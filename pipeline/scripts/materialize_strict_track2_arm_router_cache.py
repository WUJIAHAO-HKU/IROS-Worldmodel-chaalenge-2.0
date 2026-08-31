#!/usr/bin/env python3
"""Attach deployment-time arm routes to an immutable prediction cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_strict_track2_autoregressive_candidate import Windows
from wam_pipeline.arm_router import LinearVisualActionArmRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite routed cache: {args.output}")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    dataset = Windows(args.windows, split["validation_episodes"])
    by_name = {path.name: index for index, path in enumerate(dataset.paths)}
    with np.load(args.cache, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    router = LinearVisualActionArmRouter(args.router)
    instruction_doc = json.loads(args.instruction_map.read_text(encoding="utf-8"))
    instruction_by_seed = instruction_doc.get("seed_to_instruction", instruction_doc)
    routes, instructions = [], []
    for position, name in enumerate(arrays["path"].astype(str)):
        if name not in by_name:
            raise ValueError(f"cached window is missing from dataset: {name}")
        context, history, future, target, _, _ = dataset[by_name[name]]
        if not np.array_equal(context[-1], arrays["context_last"][position]):
            raise ValueError(f"context mismatch: {name}")
        if not np.array_equal(target, arrays["target"][position]):
            raise ValueError(f"target mismatch: {name}")
        seed = str(int(arrays["synthetic_seed"][position]))
        if seed not in instruction_by_seed:
            raise ValueError(f"instruction map is missing seed {seed}")
        instruction = str(instruction_by_seed[seed])
        instructions.append(instruction)
        routes.append(bool(router.predict(context, history, future, instruction)))
    arrays["route_right"] = np.asarray(routes, dtype=np.bool_)
    arrays["instruction"] = np.asarray(instructions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    target = arrays["arm_right"].astype(bool)
    route = arrays["route_right"]
    report = {
        "format": "strict-track2-arm-routed-prediction-cache-v1",
        "source_cache": str(args.cache.resolve()),
        "router": str(args.router.resolve()),
        "instruction_map": str(args.instruction_map.resolve()),
        "samples": len(route),
        "route_accuracy": float((route == target).mean()),
        "confusion": [
            [int(np.count_nonzero((target == truth) & (route == guess))) for guess in (False, True)]
            for truth in (False, True)
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
