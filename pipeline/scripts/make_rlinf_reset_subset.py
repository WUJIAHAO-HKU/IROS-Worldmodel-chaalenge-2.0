#!/usr/bin/env python3
"""Build a reproducible RLinf reset-state subset without copying large arrays."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-key", default="train_episodes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("hardlink", "copy"), default="hardlink",
        help="Hardlinks avoid duplicating reset arrays on the same filesystem.",
    )
    return parser.parse_args()


def build_subset(
    source: Path,
    split_manifest: Path,
    split_key: str,
    output: Path,
    mode: str = "hardlink",
) -> dict:
    source = source.resolve()
    split_manifest = split_manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)

    split_data = json.loads(split_manifest.read_text())
    if split_key not in split_data:
        raise KeyError(f"missing split key {split_key!r} in {split_manifest}")
    episode_ids = sorted({int(value) for value in split_data[split_key]})
    if not episode_ids:
        raise ValueError(f"split {split_key!r} is empty")

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    records_by_id = {
        int(Path(record["reset_file"]).stem.removeprefix("episode")): record
        for record in source_manifest["episodes"]
    }
    selected_records = []
    for episode_id in episode_ids:
        filename = f"episode{episode_id}.npy"
        src = source / filename
        dst = output / filename
        if not src.is_file():
            raise FileNotFoundError(src)
        if episode_id not in records_by_id:
            raise KeyError(f"episode {episode_id} missing from source manifest")
        if dst.exists():
            if not dst.samefile(src) and mode == "hardlink":
                raise FileExistsError(f"existing output is not a hardlink to source: {dst}")
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        selected_records.append(records_by_id[episode_id])

    expected = {f"episode{episode_id}.npy" for episode_id in episode_ids}
    unexpected = sorted(path.name for path in output.glob("episode*.npy") if path.name not in expected)
    if unexpected:
        raise RuntimeError(f"output contains episodes outside requested split: {unexpected}")

    manifest = {
        "format": "rlinf-reset-subset-v1",
        "source": str(source),
        "source_manifest": str(source_manifest_path),
        "split_manifest": str(split_manifest),
        "split_key": split_key,
        "mode": mode,
        "episode_ids": episode_ids,
        "episodes": selected_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = build_subset(
        args.source, args.split_manifest, args.split_key, args.output, args.mode
    )
    print(f"created {len(manifest['episode_ids'])} reset episodes in {args.output}")


if __name__ == "__main__":
    main()
