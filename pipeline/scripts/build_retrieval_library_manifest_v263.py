#!/usr/bin/env python3
"""Build the minimal episode manifest consumed by the v14 retrieval evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-key", default="train_episodes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text())
    episodes = {int(value) for value in split[args.split_key]}
    names = sorted(
        path.name for path in Path(args.windows).glob("*.npz")
        if int(path.name.split("_")[0][7:]) in episodes
    )
    if not names:
        raise RuntimeError("no training windows found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, windows=np.asarray(names))
    print(json.dumps({"episodes": len(episodes), "windows": len(names), "output": str(output)}))


if __name__ == "__main__":
    main()
