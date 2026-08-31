#!/usr/bin/env python3
"""Encode public RoboTwin episodes into the official Wan ``RLinfNpyDataset`` layout.

The Wan trainer samples its own 13-frame windows.  This encoder deliberately
writes each source episode once, instead of expanding it into overlapping
Track-2 windows, so the original temporal/action alignment is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

# Keep this executable runnable as ``python pipeline/scripts/...``.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_pipeline.data import load_robotwin_hdf5
from wam_pipeline.profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-robotwin-wan-rlinf-npy-v1"
SOURCE_NAME = "public_adjust_bottle"
DATASET_SPLIT_NAMES = {"train": "train", "validation": "val"}


def episode_id(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode") or not stem[7:].isdigit():
        raise ValueError(f"expected episode<ID>.hdf5, got {path.name}")
    return int(stem[7:])


def atomic_save(path: Path, value: np.ndarray) -> None:
    """Write one NPY atomically so interrupted encodes are never accepted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, value, allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def input_data_directory(value: Path) -> Path:
    directory = value / "data" if (value / "data").is_dir() else value
    if not directory.is_dir():
        raise ValueError(f"input must be a dataset root or data directory: {value}")
    return directory


def split_episodes(split: dict) -> tuple[set[int], set[int], set[int]]:
    keys = ("train_episodes", "validation_episodes", "local_test_episodes")
    values = []
    for key in keys:
        episode_list = split.get(key)
        if not isinstance(episode_list, list) or not all(isinstance(item, int) for item in episode_list):
            raise ValueError(f"split manifest has invalid {key}")
        values.append(set(episode_list))
    train, validation, local_test = values
    if train & validation or train & local_test or validation & local_test:
        raise ValueError("split manifest episode sets overlap")
    if not train or not validation:
        raise ValueError("Wan training needs non-empty train and validation episode sets")
    return train, validation, local_test


def destination(output: Path, split: str, episode: int) -> Path:
    try:
        directory_name = DATASET_SPLIT_NAMES[split]
    except KeyError as exc:
        raise ValueError(f"unknown logical split: {split}") from exc
    return output / f"{directory_name}_data" / SOURCE_NAME / f"episode{episode:04d}"


def encode_episode(source: Path, target: Path, resize_to: int) -> dict:
    trajectory = load_robotwin_hdf5(source, resize_to=resize_to)
    if len(trajectory.frames) < CONTEXT_FRAMES + PREDICTION_FRAMES:
        raise ValueError(f"{source.name} has fewer than {CONTEXT_FRAMES + PREDICTION_FRAMES} frames")
    if trajectory.actions.shape != (len(trajectory.frames), ACTION_DIM):
        raise ValueError(f"{source.name} does not have aligned [T,{ACTION_DIM}] actions")

    # This [T, 1, H, W, 3] / [T, 1, 14] shape is the public RLinf Wan contract.
    rgb = np.ascontiguousarray(trajectory.frames[:, None], dtype=np.uint8)
    actions = np.ascontiguousarray(trajectory.actions[:, None], dtype=np.float32)
    atomic_save(target / "rgb.npy", rgb)
    atomic_save(target / "actions.npy", actions)
    return {
        "episode": episode_id(source),
        "source": str(source.resolve()),
        "destination": str(target.resolve()),
        "frames": int(len(rgb)),
        "rgb_shape": list(rgb.shape),
        "actions_shape": list(actions.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode official Adjust Bottle trajectories for Wan training.")
    parser.add_argument("--input", required=True, help="Official data root or its data/ directory")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true", help="Re-encode existing selected episode directories.")
    args = parser.parse_args()
    if args.resize < 32:
        raise SystemExit("--resize must be at least 32")

    data_directory = input_data_directory(Path(args.input))
    source_root = data_directory.parent
    split_path = Path(args.split_manifest)
    split = json.loads(split_path.read_text())
    train, validation, local_test = split_episodes(split)
    source_by_id = {episode_id(path): path for path in sorted(data_directory.glob("episode*.hdf5"))}
    required = train | validation | local_test
    missing = sorted(required - set(source_by_id))
    if missing:
        raise SystemExit(f"source data is missing split episodes: {missing}")

    output = Path(args.output)
    encoded: dict[str, list[dict]] = {"train": [], "validation": []}
    for label, episodes in (("train", train), ("validation", validation)):
        for identifier in sorted(episodes):
            target = destination(output, label, identifier)
            rgb_path, action_path = target / "rgb.npy", target / "actions.npy"
            if rgb_path.is_file() and action_path.is_file() and not args.overwrite:
                rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
                actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
                expected = (len(rgb), 1, args.resize, args.resize, 3)
                if rgb.dtype != np.uint8 or rgb.shape != expected or actions.dtype != np.float32 or actions.shape != (len(rgb), 1, ACTION_DIM):
                    raise SystemExit(f"existing encoded episode is malformed; re-run with --overwrite: {target}")
                encoded[label].append(
                    {"episode": identifier, "source": str(source_by_id[identifier].resolve()), "destination": str(target.resolve()),
                     "frames": int(len(rgb)), "rgb_shape": list(rgb.shape), "actions_shape": list(actions.shape), "reused": True}
                )
                continue
            encoded[label].append(encode_episode(source_by_id[identifier], target, args.resize))

    manifest = {
        "format": FORMAT,
        "task": "adjust_bottle",
        "source_dataset": str(source_root.resolve()),
        "split_manifest": str(split_path.resolve()),
        "source_name": SOURCE_NAME,
        "image_size": [args.resize, args.resize],
        "context_frames": CONTEXT_FRAMES,
        "prediction_frames": PREDICTION_FRAMES,
        "wan_num_frames": CONTEXT_FRAMES + PREDICTION_FRAMES,
        "action_dim": ACTION_DIM,
        "action_representation": "robotwin_aloha_agilex_abs14",
        "excluded_local_test_episodes": sorted(local_test),
        "dataset_split_directories": {key: f"{value}_data" for key, value in DATASET_SPLIT_NAMES.items()},
        "splits": encoded,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    print(json.dumps({"status": "complete", "output": str(output.resolve()), "train_episodes": len(encoded["train"]), "validation_episodes": len(encoded["validation"]), "excluded_local_test_episodes": len(local_test)}, indent=2))


if __name__ == "__main__":
    main()
