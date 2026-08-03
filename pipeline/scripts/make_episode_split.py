#!/usr/bin/env python3
"""Create a fixed, episode-disjoint split for prepared Track 2 windows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


EPISODE_PATTERN = re.compile(r"^episode(\d+)_\d{5}\.npz$")


def episode_id(path: Path) -> int:
    match = EPISODE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"window name must be episode<id>_<start>.npz: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=5)
    args = parser.parse_args()
    paths = sorted(Path(args.windows).glob("episode*_*.npz"))
    if not paths:
        raise SystemExit("no prepared window files found")
    counts = Counter(episode_id(path) for path in paths)
    episodes = sorted(counts)
    test_episodes = len(episodes) - args.train_episodes - args.val_episodes
    if args.train_episodes < 1 or args.val_episodes < 1 or test_episodes < 1:
        raise SystemExit("train, validation, and local-test splits each need at least one episode")
    shuffled = np.random.default_rng(args.seed).permutation(episodes).tolist()
    train = sorted(shuffled[: args.train_episodes])
    validation = sorted(shuffled[args.train_episodes : args.train_episodes + args.val_episodes])
    local_test = sorted(shuffled[args.train_episodes + args.val_episodes :])
    manifest = {
        "format": "track2-episode-split-v1",
        "seed": args.seed,
        "windows_dir": str(Path(args.windows).resolve()),
        "train_episodes": train,
        "validation_episodes": validation,
        "local_test_episodes": local_test,
        "train_windows": sum(counts[episode] for episode in train),
        "validation_windows": sum(counts[episode] for episode in validation),
        "local_test_windows": sum(counts[episode] for episode in local_test),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
