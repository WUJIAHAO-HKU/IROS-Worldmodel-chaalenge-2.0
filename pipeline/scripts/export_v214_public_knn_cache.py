#!/usr/bin/env python3
"""Export a stateless public kNN world-model cache for a frozen long128 audit."""

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


def parse_name(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    episode_text, start_text = stem.split("_")
    return int(episode_text[7:]), int(start_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--query-windows", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=2.5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.chunks < 1 or min(args.visual_weight, args.action_weight) <= 0:
        raise SystemExit("chunks and retrieval weights must be positive")
    with np.load(args.baseline_cache, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        baseline = values["baseline"].copy()
        arm_right = values["arm_right"].astype(bool)
    with np.load(args.library_index, allow_pickle=False) as values:
        library_paths = values["path"].astype(str)
        library_episode = values["episode_id"].astype(np.int64)
        library_visual = values["visual"].astype(np.float32)
        library_action = values["action"].astype(np.float32)
        mean = values["normalization_mean"].astype(np.float32)
        std = values["normalization_std"].astype(np.float32)
    query_index: dict[tuple[int, int], Path] = {}
    for path in args.query_windows.glob("episode*_*.npz"):
        query_index[parse_name(path.name)] = path
    target_cache: dict[int, np.ndarray] = {}
    selected_rows, predictions = [], []
    for position, (name, is_right) in enumerate(zip(paths, arm_right), 1):
        if not is_right:
            predictions.append(baseline[position - 1])
            selected_rows.append([])
            continue
        episode, start = parse_name(name)
        first = query_index[(episode, start)]
        with np.load(first, allow_pickle=False) as values:
            context = values["context_frames"].copy()
            history = values["history_actions"].astype(np.float32).copy()
        chunks, row = [], []
        allowed = library_episode != episode
        if not allowed.any():
            raise ValueError(f"leave-one-episode-out library empty for episode {episode}")
        for chunk in range(args.chunks):
            query_path = query_index[(episode, start + 8 * chunk)]
            with np.load(query_path, allow_pickle=False) as values:
                future = values["future_actions"].astype(np.float32).copy()
            query_visual = visual_descriptor(context[-1])
            query_action = ((np.concatenate((history, future), 0) - mean) / std).reshape(-1)
            visual_distance = ((library_visual - query_visual) ** 2).mean(1)
            action_distance = ((library_action - query_action) ** 2).mean(1)
            visual_scale = max(float(np.median(visual_distance[allowed])), 1e-9)
            action_scale = max(float(np.median(action_distance[allowed])), 1e-9)
            score = args.visual_weight * visual_distance / visual_scale
            score += args.action_weight * action_distance / action_scale
            score[~allowed] = np.inf
            selected = int(np.argmin(score))
            target = target_cache.get(selected)
            if target is None:
                with np.load(library_paths[selected], allow_pickle=False) as values:
                    target = values["target_frames"].copy()
                target_cache[selected] = target
            chunks.append(target)
            row.append(
                {
                    "chunk": chunk,
                    "library_path": library_paths[selected],
                    "visual_distance": float(visual_distance[selected]),
                    "action_distance": float(action_distance[selected]),
                    "normalized_score": float(score[selected]),
                }
            )
            context = np.concatenate((context, target), 0)[-5:]
            history = np.concatenate((history, future), 0)[-4:]
        predictions.append(np.concatenate(chunks, 0))
        selected_rows.append(row)
        print(json.dumps({"exported": position, "total": len(paths), "path": name}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, candidate=np.stack(predictions), path=paths)
    manifest = {
        "format": "strict-track2-v214-public-right-knn-candidate-v1",
        "baseline_cache": str(args.baseline_cache.resolve()),
        "baseline_cache_sha256": sha256(args.baseline_cache),
        "query_windows": str(args.query_windows.resolve()),
        "library_index": str(args.library_index.resolve()),
        "library_index_sha256": sha256(args.library_index),
        "chunks": args.chunks,
        "visual_weight": args.visual_weight,
        "action_weight": args.action_weight,
        "left_route": "frozen baseline",
        "right_route": "leave-query-episode-out nearest public training window",
        "retrieval_uses_outcome_labels": False,
        "query_target_frames_read": False,
        "hidden_or_final_data": False,
        "real_submission": False,
        "selections": selected_rows,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "selections"}, indent=2))


if __name__ == "__main__":
    main()
