#!/usr/bin/env python3
"""Build an audited hard-linked mixture of demonstrations and on-policy windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


def episode_id(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if not match:
        raise ValueError(f"invalid Track 2 window name: {path.name}")
    return int(match.group(1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_arm(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "arm_right" in data.files:
            return "right" if bool(data["arm_right"]) else "left"
        actions = np.concatenate((data["history_actions"], data["future_actions"]))
    delta = np.abs(np.diff(actions, axis=0))
    return "right" if delta[:, 7:].mean() > delta[:, :7].mean() else "left"


def selected(root: Path, episodes: list[int]) -> list[Path]:
    allowed = set(int(value) for value in episodes)
    paths = [path for path in sorted(root.glob("episode*_*.npz")) if episode_id(path) in allowed]
    if not paths:
        raise ValueError(f"no windows selected from {root}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-windows", required=True)
    parser.add_argument("--official-split", required=True)
    parser.add_argument("--synthetic-windows", required=True)
    parser.add_argument("--synthetic-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--right-repeat-factor", type=int, default=2)
    args = parser.parse_args()
    if args.right_repeat_factor < 1:
        raise ValueError("right-repeat-factor must be at least one")

    official_root = Path(args.official_windows).resolve()
    synthetic_root = Path(args.synthetic_windows).resolve()
    official_split_path = Path(args.official_split).resolve()
    synthetic_split_path = Path(args.synthetic_split).resolve()
    official_split = json.loads(official_split_path.read_text())
    synthetic_split = json.loads(synthetic_split_path.read_text())
    official_train = set(int(value) for value in official_split["train_episodes"])
    official_validation = set(int(value) for value in official_split["validation_episodes"])
    synthetic_train = set(int(value) for value in synthetic_split["train_episodes"])
    synthetic_validation = set(int(value) for value in synthetic_split["validation_episodes"])
    if official_train & official_validation or synthetic_train & synthetic_validation:
        raise ValueError("source train/validation episode overlap")
    if (official_train | official_validation) & (synthetic_train | synthetic_validation):
        raise ValueError("official and synthetic episode IDs overlap")

    groups = {
        ("official", "train"): selected(official_root, sorted(official_train)),
        ("official", "validation"): selected(official_root, sorted(official_validation)),
        ("synthetic", "train"): selected(synthetic_root, sorted(synthetic_train)),
        ("synthetic", "validation"): selected(synthetic_root, sorted(synthetic_validation)),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)

    records: list[dict[str, object]] = []
    counts = Counter()
    digest_cache: dict[Path, str] = {}
    try:
        for (source, split), paths in groups.items():
            for path in paths:
                arm = active_arm(path)
                digest = digest_cache.setdefault(path, sha256_file(path))
                replicas = args.right_repeat_factor if split == "train" and arm == "right" else 1
                for replica in range(replicas):
                    suffix = "" if replica == 0 else f"_rightrep{replica}"
                    target = staging / f"{path.stem}{suffix}.npz"
                    if target.exists():
                        raise FileExistsError(f"window collision: {target.name}")
                    os.link(path, target)
                    records.append({
                        "path": target.name,
                        "source_path": str(path),
                        "source": source,
                        "split": split,
                        "episode": episode_id(path),
                        "arm": arm,
                        "replica": replica,
                        "sha256": digest,
                    })
                    counts[f"effective_{source}_{split}_{arm}_windows"] += 1
                counts[f"unique_{source}_{split}_{arm}_windows"] += 1

        train_episodes = sorted(official_train | synthetic_train)
        validation_episodes = sorted(official_validation | synthetic_validation)
        if set(train_episodes) & set(validation_episodes):
            raise AssertionError("joint split is not episode-disjoint")
        merkle = hashlib.sha256()
        for record in records:
            merkle.update(f"{record['path']}\0{record['sha256']}\n".encode())
        manifest = {
            "format": "strict-track2-joint-parent-window-mixture-v1",
            "purpose": "world-model parent training only",
            "official_data_role": "successful task demonstrations",
            "synthetic_data_role": "official-Pi0.5 on-policy success/failure dynamics",
            "real_acceptance_seeds_used": False,
            "policy_success_used_for_split_selection": False,
            "right_repeat_factor_train_only": args.right_repeat_factor,
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "train_windows": sum(1 for record in records if record["split"] == "train"),
            "validation_windows": sum(1 for record in records if record["split"] == "validation"),
            "counts": dict(sorted(counts.items())),
            "sources": {
                "official_windows": str(official_root),
                "official_split": str(official_split_path),
                "official_split_sha256": sha256_file(official_split_path),
                "synthetic_windows": str(synthetic_root),
                "synthetic_split": str(synthetic_split_path),
                "synthetic_split_sha256": sha256_file(synthetic_split_path),
            },
            "window_manifest_sha256": merkle.hexdigest(),
        }
        (staging / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "window_sources.json").write_text(json.dumps(records, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        # Hard links do not own their source data, but leave an explicit partial
        # directory for forensic inspection instead of silently reusing it.
        raise
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
