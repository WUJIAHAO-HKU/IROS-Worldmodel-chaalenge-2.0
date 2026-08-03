#!/usr/bin/env python3
"""Create deterministic API-aligned sample data when official data is not mounted yet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wam_pipeline.data import Trajectory, make_windows, write_window_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/toy_windows")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if args.frames < 13:
        raise SystemExit("--frames must be at least 13")
    y, x = np.mgrid[0:256, 0:256]
    frames = np.empty((args.frames, 256, 256, 3), dtype=np.uint8)
    for index in range(args.frames):
        frames[index, :, :, 0] = (x + 3 * index) % 256
        frames[index, :, :, 1] = (y + 5 * index) % 256
        frames[index, :, :, 2] = ((x // 2 + y // 2) + 7 * index) % 256
    actions = np.linspace(-0.5, 0.5, args.frames * 14, dtype=np.float32).reshape(args.frames, 14)
    windows = make_windows(Trajectory(frames, actions, "synthetic-toy"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, window in enumerate(windows):
        write_window_npz(window, output / f"toy_{index:05d}.npz")
    print(f"wrote {len(windows)} toy windows to {output}")


if __name__ == "__main__":
    main()
