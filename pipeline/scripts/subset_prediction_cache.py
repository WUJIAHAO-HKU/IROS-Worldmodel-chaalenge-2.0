#!/usr/bin/env python3
"""Align and subset a prediction cache to the window order of a reference cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as ref:
        names = ref["windows"].astype(str)
    with np.load(args.source, allow_pickle=False) as source:
        source_names = source["windows"].astype(str)
        lookup = {name: index for index, name in enumerate(source_names)}
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} reference windows are absent; first={missing[0]}")
        prediction = source["prediction"][[lookup[name] for name in names]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction, windows=names)
    print({"samples": len(names), "shape": prediction.shape, "output": str(output)})


if __name__ == "__main__":
    main()
