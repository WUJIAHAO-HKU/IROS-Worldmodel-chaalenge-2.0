#!/usr/bin/env python3
"""Convert official RoboTwin HDF5 episodes into API-aligned 5+8 NPZ windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from wam_pipeline.data import load_robotwin_hdf5, make_windows, write_window_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="One episode*.hdf5 or a directory containing them")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-episode", type=int, default=0, help="0 writes every valid window (default)")
    args = parser.parse_args()
    source = Path(args.input)
    episodes = [source] if source.is_file() else sorted(source.rglob("episode*.hdf5"))
    if not episodes:
        raise SystemExit("no episode*.hdf5 files found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode in episodes:
        windows = make_windows(load_robotwin_hdf5(episode))
        if args.limit_per_episode:
            windows = windows[: args.limit_per_episode]
        for local_index, window in enumerate(windows):
            write_window_npz(window, output / f"{episode.stem}_{local_index:05d}.npz")
            count += 1
    print(f"wrote {count} Track-2-aligned windows to {output}")


if __name__ == "__main__":
    main()
