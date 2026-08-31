#!/usr/bin/env python3
"""Verify all train-only labels before a formal multi-source training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def train_window_paths(windows: Path, split: dict) -> list[Path]:
    episodes = split.get("train_episodes")
    if not isinstance(episodes, list) or len(set(episodes)) != len(episodes):
        raise ValueError("split manifest has invalid train episodes")
    for key in ("validation_episodes", "local_test_episodes"):
        values = split.get(key)
        if not isinstance(values, list) or set(episodes).intersection(values):
            raise ValueError("episode splits are missing or overlap")
    paths = [
        path
        for path in sorted(windows.glob("episode*_*.npz"))
        if int(path.name.split("_")[0][7:]) in set(episodes)
    ]
    expected = int(split.get("train_windows", -1))
    if expected != len(paths):
        raise ValueError(f"train window count is {len(paths)}, expected {expected}")
    return paths


def validate_inputs(
    windows: Path,
    split_manifest: Path,
    flow_targets: Path,
    flow_resolution: int,
) -> dict:
    """Check split provenance and every memory-mapped RAFT tensor header."""
    split = json.loads(split_manifest.read_text())
    paths = train_window_paths(windows, split)
    manifest_path = flow_targets / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("missing multi-source RAFT manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "track2-raft-backward-flow-targets-v2":
        raise ValueError("RAFT labels are not multi-source v2 targets")
    if manifest.get("split") != "train_episodes_only" or manifest.get("source_mode") != "all-context":
        raise ValueError("RAFT labels are not restricted to train episodes and all context sources")
    if int(manifest.get("source_count", -1)) != 5 or int(manifest.get("flow_resolution", -1)) != flow_resolution:
        raise ValueError("RAFT source count or resolution does not match training")
    if int(manifest.get("window_count", -1)) != len(paths):
        raise ValueError("RAFT manifest count does not match the train split")
    if Path(manifest.get("source_windows", "")).resolve() != windows.resolve():
        raise ValueError("RAFT labels come from another windows directory")
    if Path(manifest.get("split_manifest", "")).resolve() != split_manifest.resolve():
        raise ValueError("RAFT labels come from another split manifest")

    expected_names = {f"{path.stem}.npy" for path in paths}
    actual_names = {path.name for path in flow_targets.glob("*.npy")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"RAFT label set mismatch: missing={len(missing)} extra={len(extra)}")
    expected_shape = (8, 5, 2, flow_resolution, flow_resolution)
    for index, path in enumerate(paths, start=1):
        value = np.load(flow_targets / f"{path.stem}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != expected_shape or value.dtype != np.float16:
            raise ValueError(
                f"invalid RAFT target {path.stem}: shape={value.shape} dtype={value.dtype}, "
                f"expected={expected_shape} float16"
            )
        del value
    return {
        "format": "track2-multisource-training-input-validation-v1",
        "status": "valid",
        "windows": str(windows.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "flow_targets": str(flow_targets.resolve()),
        "train_window_count": len(paths),
        "flow_resolution": flow_resolution,
        "source_count": 5,
        "label_dtype": "float16",
        "label_shape": list(expected_shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all multi-source Track 2 training inputs.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--flow-resolution", type=int, default=128)
    parser.add_argument("--report", help="Optional JSON evidence written atomically after a successful check.")
    args = parser.parse_args()
    if args.flow_resolution < 8:
        raise SystemExit("--flow-resolution must be at least 8")
    try:
        result = validate_inputs(
            Path(args.windows), Path(args.split_manifest), Path(args.flow_targets), args.flow_resolution
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"multi-source training input validation failed: {exc}") from exc
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
