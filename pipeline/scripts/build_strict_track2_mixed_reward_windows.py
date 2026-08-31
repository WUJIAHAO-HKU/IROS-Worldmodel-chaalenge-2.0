#!/usr/bin/env python3
"""Hard-link on-policy failures and official successful demos for reward training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_id(path: Path) -> int:
    return int(path.name.split("_")[0][7:])


def inferred_right(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as values:
        actions = np.concatenate((values["history_actions"], values["future_actions"]), 0)
    delta = np.abs(np.diff(actions, axis=0))
    return bool(delta[:, 7:].mean() > delta[:, :7].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onpolicy-windows", required=True, type=Path)
    parser.add_argument("--onpolicy-split", required=True, type=Path)
    parser.add_argument("--official-windows", required=True, type=Path)
    parser.add_argument("--official-reset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-offset", type=int, default=20000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    onpolicy_split = json.loads(args.onpolicy_split.read_text())
    official = json.loads(args.official_reset_manifest.read_text())
    official_ids = {int(value) for value in official["episode_ids"]}
    instruction = {
        int(row["episode"][7:-5]): str(row["instruction"])
        for row in official["episodes"]
    }

    onpolicy_train = {int(value) for value in onpolicy_split["train_episodes"]}
    linked_onpolicy = 0
    for source in sorted(args.onpolicy_windows.glob("episode*_*.npz")):
        if episode_id(source) in onpolicy_train:
            os.link(source.resolve(), output / source.name)
            linked_onpolicy += 1

    official_counts = {"left_success_windows": 0, "right_success_windows": 0}
    linked_official = 0
    episode_to_instruction = {}
    for source in sorted(args.official_windows.glob("episode*_*.npz")):
        original = episode_id(source)
        if original not in official_ids:
            continue
        renamed = source.name.replace(f"episode{original}_", f"episode{args.official_offset + original}_", 1)
        os.link(source.resolve(), output / renamed)
        linked_official += 1
        side = "right_success_windows" if inferred_right(source) else "left_success_windows"
        official_counts[side] += 1
        episode_to_instruction[str(args.official_offset + original)] = instruction[original]

    manifest = {
        "format": "strict-track2-mixed-reward-training-windows-v1",
        "mode": "hardlink",
        "data_use": "world-model reward alignment only; never policy BC or evaluation",
        "onpolicy_windows": str(args.onpolicy_windows.resolve()),
        "onpolicy_split": str(args.onpolicy_split.resolve()),
        "onpolicy_split_sha256": sha256(args.onpolicy_split.resolve()),
        "official_windows": str(args.official_windows.resolve()),
        "official_reset_manifest": str(args.official_reset_manifest.resolve()),
        "official_reset_manifest_sha256": sha256(args.official_reset_manifest.resolve()),
        "official_offset": args.official_offset,
        "train_episodes": sorted(onpolicy_train | {args.official_offset + value for value in official_ids}),
        "validation_episodes": onpolicy_split["validation_episodes"],
        "episode_to_instruction": episode_to_instruction,
        "linked_onpolicy_train_windows": linked_onpolicy,
        "linked_official_success_windows": linked_official,
        "official_success_arm_counts": official_counts,
        "real_development_or_acceptance_data": False,
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "linked_onpolicy_train_windows", "linked_official_success_windows", "official_success_arm_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
