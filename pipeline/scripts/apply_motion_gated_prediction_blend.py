#!/usr/bin/env python3
"""Apply a fixed observable-context motion gate to two aligned prediction caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sweep_motion_gated_prediction_blend import context_motion


def load(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"prediction", "windows"}:
            raise ValueError(f"invalid cache fields: {path}")
        return cache["prediction"], [str(name) for name in cache["windows"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--context-motion-statistic", choices=("mean", "last"), required=True)
    parser.add_argument("--context-motion-threshold", type=float, required=True)
    parser.add_argument("--low-motion-first-weight", type=float, required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.low_motion_first_weight <= 1.0 or args.context_motion_threshold < 0.0:
        raise SystemExit("weight must be in [0,1] and threshold must be non-negative")
    first, first_names = load(Path(args.first_cache))
    second, second_names = load(Path(args.second_cache))
    if first_names != second_names or first.shape != second.shape:
        raise ValueError("prediction caches are not aligned")
    contexts = []
    for name in first_names:
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            contexts.append(window["context_frames"])
    observed = context_motion(np.stack(contexts), args.context_motion_statistic)
    use_first = observed >= args.context_motion_threshold
    prediction = first.copy()
    prediction[~use_first] = np.rint(
        first[~use_first].astype(np.float32) * args.low_motion_first_weight
        + second[~use_first].astype(np.float32) * (1.0 - args.low_motion_first_weight)
    ).clip(0, 255).astype(np.uint8)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, prediction=prediction, windows=np.asarray(first_names))
    result = {
        "format": "track2-applied-motion-gated-prediction-blend-v1",
        "first_cache": str(Path(args.first_cache).resolve()),
        "second_cache": str(Path(args.second_cache).resolve()),
        "sample_count": len(first_names),
        "context_motion_statistic": args.context_motion_statistic,
        "context_motion_threshold": args.context_motion_threshold,
        "low_motion_first_weight": args.low_motion_first_weight,
        "low_motion_second_weight": 1.0 - args.low_motion_first_weight,
        "use_first_count": int(use_first.sum()),
        "use_blend_count": int((~use_first).sum()),
        "context_motion_min": float(observed.min()),
        "context_motion_mean": float(observed.mean()),
        "context_motion_max": float(observed.max()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
