#!/usr/bin/env python3
"""Cross-validate RGB blends from two audited prediction caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("samples must be positive")
    if count >= len(paths):
        return paths
    return [paths[index] for index in np.linspace(0, len(paths) - 1, count, dtype=np.int64)]


def load_cache(path: Path, names: list[str], shape: tuple[int, ...]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"prediction", "windows"}:
            raise ValueError(f"invalid prediction cache fields: {path}")
        prediction = data["prediction"]
        cached_names = data["windows"].tolist()
    if prediction.shape != shape or prediction.dtype != np.uint8:
        raise ValueError(f"invalid prediction cache array: {path}: {prediction.shape} {prediction.dtype}")
    if cached_names != names:
        raise ValueError(f"prediction cache window ordering mismatch: {path}")
    return prediction


def motion_mask(context: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    previous = np.concatenate((context[:, -1:], target[:, :-1]), axis=1)
    return np.abs(target.astype(np.float32) - previous.astype(np.float32)).mean(axis=(1, 2, 3, 4)) / 255.0 >= threshold


def metrics(error: np.ndarray, selected: np.ndarray, high_motion: np.ndarray) -> dict:
    chosen = error[selected]
    high = selected & high_motion
    return {
        "sample_count": int(selected.sum()),
        "mae_mean": float(chosen.mean()),
        "mae_by_prediction_frame": [float(value) for value in chosen.mean(axis=0)],
        "high_motion_sample_count": int(high.sum()),
        "high_motion_mae_mean": float(error[high].mean()) if high.any() else None,
        "max_window_prediction_frame_mae": float(chosen.max()),
    }


def prediction_error(prediction: np.ndarray, target: np.ndarray, chunk_size: int = 16) -> np.ndarray:
    """Return per-window/per-horizon MAE without a full float32 video copy."""
    error = np.empty(prediction.shape[:2], dtype=np.float32)
    for start in range(0, len(prediction), chunk_size):
        stop = min(start + chunk_size, len(prediction))
        difference = prediction[start:stop].astype(np.int16) - target[start:stop].astype(np.int16)
        error[start:stop] = np.abs(difference).mean(axis=(2, 3, 4))
    return error


def blended_error(
    first: np.ndarray,
    second: np.ndarray,
    target: np.ndarray,
    first_weights: np.ndarray,
    chunk_size: int = 16,
) -> np.ndarray:
    """Blend and score small window chunks to cap temporary memory usage."""
    weights = first_weights.reshape(1, 8, 1, 1, 1).astype(np.float32)
    error = np.empty(first.shape[:2], dtype=np.float32)
    for start in range(0, len(first), chunk_size):
        stop = min(start + chunk_size, len(first))
        values = (
            first[start:stop].astype(np.float32) * weights
            + second[start:stop].astype(np.float32) * (1.0 - weights)
        )
        prediction = np.rint(values).clip(0, 255).astype(np.uint8)
        difference = prediction.astype(np.int16) - target[start:stop].astype(np.int16)
        error[start:stop] = np.abs(difference).mean(axis=(2, 3, 4))
    return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 4 or not 0 < args.weight_step <= 1:
        raise SystemExit("samples must be at least four and weight step must be in (0,1]")

    split = json.loads(Path(args.split_manifest).read_text())
    paths = [
        path
        for episode in split["validation_episodes"]
        for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))
    ]
    selected_paths = evenly_spaced(paths, args.samples)
    names = [path.name for path in selected_paths]
    contexts, targets = [], []
    for path in selected_paths:
        with np.load(path, allow_pickle=False) as data:
            contexts.append(data["context_frames"])
            targets.append(data["target_frames"])
    context, target = np.stack(contexts), np.stack(targets)
    shape = target.shape
    first = load_cache(Path(args.first_cache), names, shape)
    second = load_cache(Path(args.second_cache), names, shape)
    high_motion = motion_mask(context, target, args.high_motion_threshold)
    tuning = np.arange(len(target)) % 2 == 0
    holdout = ~tuning
    all_samples = np.ones(len(target), dtype=bool)
    weights = np.unique(np.append(np.arange(0.0, 1.0 + args.weight_step / 2.0, args.weight_step), 1.0))

    global_candidates = []
    errors_by_weight = []
    for weight in weights:
        error = blended_error(first, second, target, np.full(8, weight))
        errors_by_weight.append(error)
        global_candidates.append({
            "first_weight": float(weight),
            "tuning_mae": metrics(error, tuning, high_motion)["mae_mean"],
        })
    selected_global = min(global_candidates, key=lambda item: item["tuning_mae"])
    global_weights = np.full(8, selected_global["first_weight"])
    global_index = int(np.flatnonzero(weights == selected_global["first_weight"])[0])
    global_error = errors_by_weight[global_index]

    first_error = prediction_error(first, target)
    second_error = prediction_error(second, target)
    horizon_weights = []
    horizon_error = np.empty(first.shape[:2], dtype=np.float32)
    for horizon in range(8):
        best_index, best_error = 0, float("inf")
        for index, error in enumerate(errors_by_weight):
            tuning_error = float(error[tuning, horizon].mean())
            if tuning_error < best_error:
                best_index, best_error = index, tuning_error
        horizon_weights.append(float(weights[best_index]))
        horizon_error[:, horizon] = errors_by_weight[best_index][:, horizon]

    result = {
        "format": "track2-crossvalidated-prediction-blend-v1",
        "first_name": args.first_name,
        "second_name": args.second_name,
        "sample_count": len(target),
        "tuning_indices": np.flatnonzero(tuning).tolist(),
        "holdout_indices": np.flatnonzero(holdout).tolist(),
        "high_motion_sample_count": int(high_motion.sum()),
        "first": {"all": metrics(first_error, all_samples, high_motion), "holdout": metrics(first_error, holdout, high_motion)},
        "second": {"all": metrics(second_error, all_samples, high_motion), "holdout": metrics(second_error, holdout, high_motion)},
        "global_blend": {
            "first_weight": float(global_weights[0]),
            "second_weight": float(1.0 - global_weights[0]),
            "all": metrics(global_error, all_samples, high_motion),
            "holdout": metrics(global_error, holdout, high_motion),
        },
        "per_horizon_blend": {
            "first_weights": horizon_weights,
            "second_weights": [float(1.0 - value) for value in horizon_weights],
            "all": metrics(horizon_error, all_samples, high_motion),
            "holdout": metrics(horizon_error, holdout, high_motion),
        },
        "diagnostic": {
            "first_better_window_frames": int((first_error < second_error).sum()),
            "second_better_window_frames": int((second_error < first_error).sum()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
