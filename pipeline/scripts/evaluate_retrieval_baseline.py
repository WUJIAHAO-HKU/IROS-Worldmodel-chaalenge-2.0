#!/usr/bin/env python3
"""Measure sharp nearest-neighbour Track 2 prediction without split leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from wam_pipeline.data import load_window_npz


def paths_for(directory: Path, episodes: list[int]) -> list[Path]:
    return [path for episode in episodes for path in sorted(directory.glob(f"episode{episode}_*.npz"))]


def image_feature(frames: np.ndarray, size: int) -> np.ndarray:
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) for frame in frames]
    ).reshape(-1) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split", choices=("validation", "local-test"), default="validation")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument(
        "--accept-mae",
        type=float,
        default=1.0,
        help="Hard acceptance threshold in 0--255 RGB MAE for every window and future frame.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    windows = Path(args.windows)
    train_paths = paths_for(windows, split["train_episodes"])
    query_key = "validation_episodes" if args.split == "validation" else "local_test_episodes"
    query_paths = paths_for(windows, split[query_key])
    selection = np.linspace(0, len(query_paths) - 1, min(args.samples, len(query_paths)), dtype=np.int64)
    query_paths = [query_paths[index] for index in selection]
    if args.accept_mae <= 0:
        raise SystemExit("--accept-mae must be positive")

    use_images = args.image_weight != 0.0
    train_images = []
    train_actions = []
    for path in train_paths:
        window = load_window_npz(path)
        if use_images:
            train_images.append(image_feature(window.context_frames, args.image_size))
        train_actions.append(np.concatenate([window.history_actions, window.future_actions]).reshape(-1))
    image_matrix = np.asarray(train_images, dtype=np.float32) if use_images else None
    action_matrix = np.asarray(train_actions, dtype=np.float32)
    action_mean, action_std = action_matrix.mean(0), action_matrix.std(0).clip(1e-5)
    action_matrix = (action_matrix - action_mean) / action_std

    errors, copy_errors, selected = [], [], []
    for path in query_paths:
        window = load_window_npz(path)
        image = image_feature(window.context_frames, args.image_size) if use_images else None
        actions = (np.concatenate([window.history_actions, window.future_actions]).reshape(-1) - action_mean) / action_std
        image_distance = np.mean(np.square(image_matrix - image), axis=1) if use_images else np.zeros(len(train_paths), dtype=np.float32)
        action_distance = np.mean(np.square(action_matrix - actions), axis=1)
        index = int(np.argmin(args.image_weight * image_distance + args.action_weight * action_distance))
        prediction = load_window_npz(train_paths[index]).target_frames
        errors.append(np.abs(prediction.astype(np.float32) - window.target_frames.astype(np.float32)).mean(axis=(1, 2, 3)))
        copy_errors.append(np.abs(window.context_frames[-1:].astype(np.float32) - window.target_frames.astype(np.float32)).mean(axis=(1, 2, 3)))
        selected.append({"query": path.name, "match": train_paths[index].name, "image_distance": float(image_distance[index]), "action_distance": float(action_distance[index])})
    per_window = np.asarray(errors, dtype=np.float32)
    per_window_mean = per_window.mean(axis=1)
    per_window_peak = per_window.max(axis=1)
    result = {
        "format": "track2-retrieval-eval-v1",
        "split": args.split,
        "sample_count": len(query_paths),
        "train_window_count": len(train_paths),
        "image_size": args.image_size,
        "image_weight": args.image_weight,
        "action_weight": args.action_weight,
        "model_mae_by_prediction_frame": np.mean(errors, axis=0).astype(float).tolist(),
        "model_mae_mean": float(np.mean(errors)),
        "copy_last_mae_mean": float(np.mean(copy_errors)),
        "hard_acceptance": {
            "threshold_mae_0_255": float(args.accept_mae),
            "window_count": int(len(per_window)),
            "windows_passing_mean_over_8_frames": int((per_window_mean < args.accept_mae).sum()),
            "windows_passing_all_8_prediction_frames": int((per_window_peak < args.accept_mae).sum()),
            "max_window_mean_over_8_frames": float(per_window_mean.max()),
            "max_window_prediction_frame_mae": float(per_window_peak.max()),
            "all_windows_and_frames_pass": bool((per_window_peak < args.accept_mae).all()),
        },
        "matches": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "matches"}, indent=2))


if __name__ == "__main__":
    main()
