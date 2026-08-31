#!/usr/bin/env python3
"""Strictly validate a Track 2 public-data-only Wan ``RLinfNpyDataset`` tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_pipeline.profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


FORMAT = "track2-robotwin-wan-rlinf-npy-v1"
SOURCE_NAME = "public_adjust_bottle"
DATASET_SPLIT_NAMES = {"train": "train", "validation": "val"}


def validate(root: Path, split_manifest: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing Wan dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("unexpected Wan dataset format")
    if manifest.get("task") != "adjust_bottle" or manifest.get("source_name") != SOURCE_NAME:
        raise ValueError("Wan dataset is not the expected Adjust Bottle source")
    if tuple(manifest.get("image_size", [])) != (256, 256):
        raise ValueError("Wan dataset image size must be [256,256]")
    if (manifest.get("context_frames"), manifest.get("prediction_frames"), manifest.get("wan_num_frames"), manifest.get("action_dim")) != (CONTEXT_FRAMES, PREDICTION_FRAMES, 13, ACTION_DIM):
        raise ValueError("Wan temporal/action contract does not match Track 2")
    if manifest.get("action_representation") != "robotwin_aloha_agilex_abs14":
        raise ValueError("Wan dataset action representation is not official abs14")
    if Path(manifest.get("split_manifest", "")).resolve() != split_manifest.resolve():
        raise ValueError("Wan dataset was built from another split manifest")
    expected_directories = {key: f"{value}_data" for key, value in DATASET_SPLIT_NAMES.items()}
    if manifest.get("dataset_split_directories") != expected_directories:
        raise ValueError("Wan dataset split directory names do not match the official RLinf layout")

    split = json.loads(split_manifest.read_text())
    expected = {"train": set(split["train_episodes"]), "validation": set(split["validation_episodes"])}
    forbidden = set(split["local_test_episodes"])
    actual: dict[str, set[int]] = {}
    frame_count: dict[str, int] = {}
    loader_episode_count: dict[str, int] = {}
    for label, episode_entries in manifest.get("splits", {}).items():
        if label not in expected or not isinstance(episode_entries, list):
            raise ValueError(f"invalid manifest split {label!r}")
        identifiers = {int(entry["episode"]) for entry in episode_entries}
        if identifiers != expected[label]:
            raise ValueError(f"{label} episodes do not exactly match the split manifest")
        actual[label] = identifiers
        count = 0
        data_directory = root / expected_directories[label]
        loader_paths = []
        for source_directory in sorted(data_directory.iterdir()):
            if not source_directory.is_dir():
                continue
            loader_paths.extend(path for path in sorted(source_directory.iterdir()) if path.is_dir())
        if len(loader_paths) != len(identifiers):
            raise ValueError(
                f"RLinfNpyDataset would discover {len(loader_paths)} {label} episodes, expected {len(identifiers)}"
            )
        for identifier in sorted(identifiers):
            directory = data_directory / SOURCE_NAME / f"episode{identifier:04d}"
            rgb_path, action_path = directory / "rgb.npy", directory / "actions.npy"
            if not rgb_path.is_file() or not action_path.is_file():
                raise ValueError(f"missing encoded files for {label} episode {identifier}")
            rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
            actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
            if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[1:] != (1, 256, 256, 3):
                raise ValueError(f"invalid rgb tensor for {label} episode {identifier}: {rgb.shape} {rgb.dtype}")
            if actions.dtype != np.float32 or actions.shape != (len(rgb), 1, ACTION_DIM) or not np.isfinite(actions).all():
                raise ValueError(f"invalid action tensor for {label} episode {identifier}: {actions.shape} {actions.dtype}")
            if len(rgb) < CONTEXT_FRAMES + PREDICTION_FRAMES:
                raise ValueError(f"episode {identifier} is too short for a 13-frame Wan window")
            count += int(len(rgb))
        frame_count[label] = count
        loader_episode_count[label] = len(loader_paths)
    if set(manifest.get("excluded_local_test_episodes", [])) != forbidden:
        raise ValueError("local-test exclusion does not match split manifest")
    if (actual["train"] | actual["validation"]) & forbidden:
        raise ValueError("local-test episode leaked into Wan train or validation data")
    return {
        "format": "track2-robotwin-wan-data-validation-v1",
        "status": "valid",
        "dataset_root": str(root.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "train_episode_count": len(actual["train"]),
        "validation_episode_count": len(actual["validation"]),
        "train_frame_count": frame_count["train"],
        "validation_frame_count": frame_count["validation"],
        "rlinf_dataset_episode_count": loader_episode_count,
        "excluded_local_test_episode_count": len(forbidden),
        "wan_window": {"context_frames": CONTEXT_FRAMES, "future_frames": PREDICTION_FRAMES, "total_frames": 13, "action_dim": ACTION_DIM},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        result = validate(Path(args.dataset_root), Path(args.split_manifest))
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"Wan dataset validation failed: {exc}") from exc
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
