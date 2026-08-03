#!/usr/bin/env python3
"""Create RLinf reset trajectories from the published Adjust Bottle HDF5 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.data import load_robotwin_hdf5


def instruction_for_episode(instruction_root: Path, episode_name: str) -> str:
    path = instruction_root / f"{episode_name}.json"
    with path.open() as handle:
        choices = json.load(handle)
    seen = choices.get("seen")
    if not isinstance(seen, list) or not seen or not isinstance(seen[0], str):
        raise ValueError(f"{path} has no usable seen instruction")
    return seen[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Published dataset root or its data/ directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional count for a short smoke dataset")
    args = parser.parse_args()

    input_root = Path(args.input)
    data_root = input_root / "data" if (input_root / "data").is_dir() else input_root
    dataset_root = data_root.parent
    instruction_root = dataset_root / "instructions"
    episodes = sorted(data_root.glob("episode*.hdf5"))
    if not episodes:
        raise SystemExit(f"no episode*.hdf5 files found in {data_root}")
    if not instruction_root.is_dir():
        raise SystemExit(f"missing public instruction directory: {instruction_root}")
    if args.limit:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        episodes = episodes[: args.limit]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for episode in episodes:
        trajectory = load_robotwin_hdf5(episode)
        if len(trajectory.frames) < 5:
            raise RuntimeError(f"{episode} has fewer than five public RGB observations")
        frames: list[dict[str, object]] = []
        instruction = instruction_for_episode(instruction_root, episode.stem)
        for index in range(5):
            frames.append(
                {
                    "image": trajectory.frames[index],
                    # WanEnv uses frame zero as a reference and its reset logic
                    # consumes the remaining four actions as temporal history.
                    "abs_action": trajectory.actions[index],
                    "instruction": instruction,
                }
            )
        target = output / f"{episode.stem}.npy"
        np.save(target, np.asarray(frames, dtype=object), allow_pickle=True)
        manifest.append(
            {
                "episode": episode.name,
                "reset_file": target.name,
                "frames": 5,
                "action_shape": [14],
                "instruction": instruction,
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(dataset_root.resolve()),
                "purpose": "public-data-only local RLinf reset states; not an organizer hidden reset set",
                "episodes": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(manifest)} public reset trajectories to {output}")


if __name__ == "__main__":
    main()
