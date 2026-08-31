#!/usr/bin/env python3
"""Build a public-data-only right-arm action/visual retrieval index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_descriptor(frame: np.ndarray) -> np.ndarray:
    value = frame.astype(np.float32).reshape(16, 16, 16, 16, 3).mean((1, 3)) / 255.0
    return value.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    split = json.loads(args.split_manifest.read_text())
    train_episodes = set(map(int, split["train_episodes"]))
    with np.load(args.normalization, allow_pickle=False) as values:
        mean = values["mean"].astype(np.float32)
        std = values["std"].astype(np.float32)
    paths, episodes, visual, action = [], [], [], []
    for path in sorted(args.windows.glob("episode*_*.npz")):
        episode = int(path.name.split("_")[0][7:])
        if episode not in train_episodes:
            continue
        with np.load(path, allow_pickle=False) as values:
            if "arm_right" in values.files:
                right = bool(values["arm_right"])
            else:
                sequence = np.concatenate((values["history_actions"], values["future_actions"]), 0)
                delta = np.abs(np.diff(sequence.astype(np.float32), axis=0))
                right = bool(delta[:, 7:].mean() > delta[:, :7].mean())
            if not right:
                continue
            context = values["context_frames"][-1]
            actions = np.concatenate((values["history_actions"], values["future_actions"]), 0).astype(np.float32)
        paths.append(str(path.resolve()))
        episodes.append(episode)
        visual.append(visual_descriptor(context))
        action.append(((actions - mean) / std).reshape(-1))
        if len(paths) == 1 or len(paths) % 512 == 0:
            print(json.dumps({"indexed": len(paths), "path": path.name}), flush=True)
    if not paths:
        raise ValueError("right-arm retrieval library is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        path=np.asarray(paths),
        episode_id=np.asarray(episodes, dtype=np.int64),
        visual=np.asarray(visual, dtype=np.float16),
        action=np.asarray(action, dtype=np.float16),
        normalization_mean=mean,
        normalization_std=std,
    )
    manifest = {
        "format": "strict-track2-v214-public-right-knn-library-v1",
        "data_policy": "declared public training episodes only",
        "selection_fields": ["context_frames", "history_actions", "future_actions", "arm_right"],
        "outcome_labels_used_for_index_or_retrieval": False,
        "windows": str(args.windows.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256(args.split_manifest),
        "normalization": str(args.normalization.resolve()),
        "normalization_sha256": sha256(args.normalization),
        "right_window_count": len(paths),
        "episode_count": len(set(episodes)),
        "visual_descriptor": "16x16 RGB block mean",
        "action_descriptor": "official-normalized flattened 4-history plus 8-future actions",
        "hidden_or_final_data": False,
        "official_requests": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
