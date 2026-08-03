#!/usr/bin/env python3
"""Report the exact public HDF5 fields needed by the Track 2 data adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing episode*.hdf5")
    args = parser.parse_args()
    episodes = sorted(Path(args.input).glob("episode*.hdf5"))
    if not episodes:
        raise SystemExit("no episode*.hdf5 files found")
    lengths = []
    for path in episodes:
        with h5py.File(path, "r") as handle:
            rgb = handle["observation/head_camera/rgb"]
            action = handle["joint_action/vector"]
            if rgb.ndim != 1 or action.ndim != 2 or action.shape != (len(rgb), 14):
                raise SystemExit(f"{path} does not provide aligned head RGB and [T,14] actions")
            if not np.isfinite(np.asarray(action)).all():
                raise SystemExit(f"{path} has non-finite action values")
            lengths.append(len(rgb))
    print(
        f"episodes={len(episodes)} min_frames={min(lengths)} max_frames={max(lengths)} "
        f"track2_windows={sum(length - 12 for length in lengths)} fields="
        "observation/head_camera/rgb,joint_action/vector[T,14]"
    )


if __name__ == "__main__":
    main()
