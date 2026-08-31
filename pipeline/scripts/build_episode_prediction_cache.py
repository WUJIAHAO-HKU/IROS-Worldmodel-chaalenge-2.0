#!/usr/bin/env python3
"""Extract one complete episode and attach its ground-truth window tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def window_start(name: str) -> int:
    match = re.search(r"_(\d+)\.npz$", name)
    if match is None:
        raise ValueError(f"cannot parse window start from {name!r}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--episode", required=True, help="For example: episode7")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        names = cache["windows"].astype(str)
        selected = np.asarray(
            [index for index, name in enumerate(names) if Path(name).stem.split("_")[0] == args.episode],
            dtype=np.int64,
        )
        if not len(selected):
            raise KeyError(f"{args.episode} is absent from {args.prediction_cache}")
        prediction = cache["prediction"][selected]
        selected_names = names[selected]

    order = np.argsort([window_start(name) for name in selected_names])
    prediction, selected_names = prediction[order], selected_names[order]
    starts = np.asarray([window_start(name) for name in selected_names])
    if len(starts) > 1 and not np.all(np.diff(starts) == 1):
        raise ValueError(f"episode windows are not consecutive: {starts.tolist()}")

    target, context = [], []
    windows = Path(args.windows)
    for name in selected_names:
        with np.load(windows / name, allow_pickle=False) as window:
            target.append(window["target_frames"].copy())
            context.append(window["context_frames"].copy())
    target_array, context_array = np.stack(target), np.stack(context)

    # Consecutive windows must represent the same underlying episode timeline.
    for index in range(1, len(selected_names)):
        if not np.array_equal(context_array[index - 1, 1:], context_array[index, :-1]):
            raise ValueError(f"context discontinuity before {selected_names[index]}")
        if not np.array_equal(target_array[index - 1, 1:], target_array[index, :-1]):
            raise ValueError(f"target discontinuity before {selected_names[index]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prediction=prediction,
        target=target_array,
        context=context_array,
        windows=selected_names,
        window_start=starts,
    )
    print(f"wrote {len(selected_names)} consecutive windows to {output}")


if __name__ == "__main__":
    main()
