#!/usr/bin/env python3
"""Measure deterministic open-loop Track 2 predictions on episode-held-out windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("--samples must be positive")
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    return [paths[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backend", choices=("ivideogpt", "residual-unet", "flow-residual-unet", "temporal-unet", "autoregressive-unet", "hybrid-unet"), default="ivideogpt")
    parser.add_argument("--split", choices=("validation", "local-test"), default="validation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    key = "validation_episodes" if args.split == "validation" else "local_test_episodes"
    episodes = split[key]
    paths = [
        path
        for episode in episodes
        for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))
    ]
    if not paths:
        raise SystemExit(f"no windows for {args.split} split")

    backend = build_backend(args.backend, args.checkpoint_dir, args.device)
    selected = evenly_spaced(paths, args.samples)
    model_maes: list[np.ndarray] = []
    copy_last_maes: list[np.ndarray] = []
    action_sensitivity: list[float] = []
    for path in selected:
        window = load_window_npz(path)
        prediction = backend.predict(
            window.context_frames,
            window.history_actions,
            window.future_actions,
            args.seed,
            instruction=None,
        )
        altered_actions = window.future_actions.copy()
        altered_actions[:, 0] += 0.05
        altered = backend.predict(
            window.context_frames,
            window.history_actions,
            altered_actions,
            args.seed,
            instruction=None,
        )
        target = window.target_frames.astype(np.float32)
        model_maes.append(np.abs(prediction.astype(np.float32) - target).mean(axis=(1, 2, 3)))
        copy_last_maes.append(np.abs(window.context_frames[-1:].astype(np.float32) - target).mean(axis=(1, 2, 3)))
        action_sensitivity.append(float(np.abs(prediction.astype(np.float32) - altered.astype(np.float32)).mean()))

    model_by_frame = np.mean(model_maes, axis=0)
    copy_by_frame = np.mean(copy_last_maes, axis=0)
    result = {
        "format": "track2-open-loop-eval-v1",
        "split": args.split,
        "episodes": episodes,
        "sample_count": len(selected),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "backend": args.backend,
        "model_mae_by_prediction_frame": [float(value) for value in model_by_frame],
        "model_mae_mean": float(model_by_frame.mean()),
        "copy_last_mae_by_prediction_frame": [float(value) for value in copy_by_frame],
        "copy_last_mae_mean": float(copy_by_frame.mean()),
        "action_perturbation_mean_absolute_pixel_delta": float(np.mean(action_sensitivity)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
