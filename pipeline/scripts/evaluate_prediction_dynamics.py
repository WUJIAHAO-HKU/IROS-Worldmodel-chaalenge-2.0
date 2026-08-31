#!/usr/bin/env python3
"""Measure temporal-motion and local-texture fidelity from a prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def laplacian(frames: np.ndarray) -> np.ndarray:
    padded = np.pad(frames, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return (
        4.0 * frames
        - padded[:, :-2, 1:-1]
        - padded[:, 2:, 1:-1]
        - padded[:, 1:-1, :-2]
        - padded[:, 1:-1, 2:]
    )


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> list[float | None]:
    return [float(n / d) if d else None for n, d in zip(numerator, denominator)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=("train", "validation", "local-test"))
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    args = parser.parse_args()
    if min(args.motion_threshold, args.high_motion_threshold) < 0.0:
        raise SystemExit("motion thresholds must be non-negative")
    if bool(args.split_manifest) != bool(args.split):
        raise SystemExit("split manifest and split must be supplied together")

    windows_dir = Path(args.windows)
    cache_path = Path(args.prediction_cache)
    cache = np.load(cache_path, allow_pickle=False)
    if set(cache.files) != {"prediction", "windows"}:
        raise ValueError(f"unexpected prediction-cache keys: {cache.files}")
    predictions = cache["prediction"]
    names = [str(name) for name in cache["windows"]]
    if predictions.ndim != 5 or predictions.shape[0] != len(names) or predictions.shape[1] != 8:
        raise ValueError(f"unexpected prediction-cache shape: {predictions.shape}")
    if len(names) != len(set(names)):
        raise ValueError("prediction cache contains duplicate windows")

    split_manifest_path = Path(args.split_manifest) if args.split_manifest else None
    if split_manifest_path is not None:
        manifest = json.loads(split_manifest_path.read_text())
        key = args.split.replace("-", "_") + "_episodes"
        allowed = {int(value) for value in manifest[key]}
        for name in names:
            match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
            if match is None or int(match.group(1)) not in allowed:
                raise ValueError(f"window {name!r} does not belong to split {args.split!r}")

    horizons = predictions.shape[1]
    rgb_sum = np.zeros(horizons, dtype=np.float64)
    rgb_count = np.zeros(horizons, dtype=np.float64)
    delta_sum = np.zeros(horizons, dtype=np.float64)
    delta_count = np.zeros(horizons, dtype=np.float64)
    laplacian_sum = np.zeros(horizons, dtype=np.float64)
    laplacian_count = np.zeros(horizons, dtype=np.float64)
    moving_rgb_sum = np.zeros(horizons, dtype=np.float64)
    moving_delta_sum = np.zeros(horizons, dtype=np.float64)
    moving_laplacian_sum = np.zeros(horizons, dtype=np.float64)
    moving_rgb_count = np.zeros(horizons, dtype=np.float64)
    moving_pixel_count = np.zeros(horizons, dtype=np.float64)
    pixel_count = np.zeros(horizons, dtype=np.float64)
    target_motion_sum = np.zeros(horizons, dtype=np.float64)
    prediction_motion_sum = np.zeros(horizons, dtype=np.float64)
    motion_magnitude_error_sum = np.zeros(horizons, dtype=np.float64)
    window_rgb_sum = np.zeros(len(names), dtype=np.float64)
    window_rgb_count = np.zeros(len(names), dtype=np.float64)
    window_future_motion = np.zeros(len(names), dtype=np.float64)

    for index, name in enumerate(names):
        path = windows_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            context_last = window["context_frames"][-1:].astype(np.float32)
        prediction = predictions[index].astype(np.float32)
        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch for {name}: {prediction.shape} versus {target.shape}")

        target_previous = np.concatenate([context_last, target[:-1]], axis=0)
        prediction_previous = np.concatenate([context_last, prediction[:-1]], axis=0)
        target_delta = target - target_previous
        prediction_delta = prediction - prediction_previous
        rgb_error = np.abs(prediction - target)
        delta_error = np.abs(prediction_delta - target_delta)
        laplacian_error = np.abs(laplacian(prediction) - laplacian(target))
        target_motion = np.abs(target_delta).mean(axis=-1)
        prediction_motion = np.abs(prediction_delta).mean(axis=-1)
        moving = target_motion >= args.motion_threshold * 255.0
        moving_rgb = np.broadcast_to(moving[..., None], rgb_error.shape)

        rgb_sum += rgb_error.sum(axis=(1, 2, 3), dtype=np.float64)
        rgb_count += np.prod(rgb_error.shape[1:])
        delta_sum += delta_error.sum(axis=(1, 2, 3), dtype=np.float64)
        delta_count += np.prod(delta_error.shape[1:])
        laplacian_sum += laplacian_error.sum(axis=(1, 2, 3), dtype=np.float64)
        laplacian_count += np.prod(laplacian_error.shape[1:])
        moving_rgb_sum += (rgb_error * moving_rgb).sum(axis=(1, 2, 3), dtype=np.float64)
        moving_delta_sum += (delta_error * moving_rgb).sum(axis=(1, 2, 3), dtype=np.float64)
        moving_laplacian_sum += (laplacian_error * moving_rgb).sum(axis=(1, 2, 3), dtype=np.float64)
        moving_rgb_count += moving.sum(axis=(1, 2), dtype=np.float64) * 3.0
        moving_pixel_count += moving.sum(axis=(1, 2), dtype=np.float64)
        pixel_count += np.prod(moving.shape[1:])
        target_motion_sum += target_motion.sum(axis=(1, 2), dtype=np.float64)
        prediction_motion_sum += prediction_motion.sum(axis=(1, 2), dtype=np.float64)
        motion_magnitude_error_sum += np.abs(prediction_motion - target_motion).sum(
            axis=(1, 2), dtype=np.float64
        )
        window_rgb_sum[index] = rgb_error.sum(dtype=np.float64)
        window_rgb_count[index] = rgb_error.size
        window_future_motion[index] = float(target_motion.mean(dtype=np.float64) / 255.0)

    def overall(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
        total = float(denominator.sum())
        return float(numerator.sum() / total) if total else None

    target_motion_by_frame = np.asarray(safe_divide(target_motion_sum, pixel_count), dtype=np.float64)
    prediction_motion_by_frame = np.asarray(safe_divide(prediction_motion_sum, pixel_count), dtype=np.float64)
    high_motion = window_future_motion >= args.high_motion_threshold
    result = {
        "format": "track2-prediction-dynamics-diagnostic-v1",
        "windows": str(windows_dir.resolve()),
        "prediction_cache": str(cache_path.resolve()),
        "prediction_cache_sha256": sha256(cache_path),
        "split_manifest": str(split_manifest_path.resolve()) if split_manifest_path else None,
        "split": args.split,
        "sample_count": len(names),
        "motion_threshold_normalized": args.motion_threshold,
        "high_motion_threshold_normalized": args.high_motion_threshold,
        "units": "uint8 intensity levels (0-255)",
        "aggregation": "pixel-weighted across all sampled windows",
        "metrics": {
            "rgb_mae_mean": overall(rgb_sum, rgb_count),
            "rgb_mae_by_prediction_frame": safe_divide(rgb_sum, rgb_count),
            "high_motion_sample_count": int(high_motion.sum()),
            "high_motion_rgb_mae_mean": float(
                window_rgb_sum[high_motion].sum() / window_rgb_count[high_motion].sum()
            ) if high_motion.any() else None,
            "temporal_delta_mae_mean": overall(delta_sum, delta_count),
            "temporal_delta_mae_by_prediction_frame": safe_divide(delta_sum, delta_count),
            "laplacian_mae_mean": overall(laplacian_sum, laplacian_count),
            "laplacian_mae_by_prediction_frame": safe_divide(laplacian_sum, laplacian_count),
            "moving_region_rgb_mae_mean": overall(moving_rgb_sum, moving_rgb_count),
            "moving_region_rgb_mae_by_prediction_frame": safe_divide(moving_rgb_sum, moving_rgb_count),
            "moving_region_temporal_delta_mae_mean": overall(moving_delta_sum, moving_rgb_count),
            "moving_region_temporal_delta_mae_by_prediction_frame": safe_divide(
                moving_delta_sum, moving_rgb_count
            ),
            "moving_region_laplacian_mae_mean": overall(moving_laplacian_sum, moving_rgb_count),
            "moving_region_laplacian_mae_by_prediction_frame": safe_divide(
                moving_laplacian_sum, moving_rgb_count
            ),
            "moving_pixel_fraction": overall(moving_pixel_count, pixel_count),
            "moving_pixel_fraction_by_prediction_frame": safe_divide(moving_pixel_count, pixel_count),
            "target_motion_magnitude_mean": overall(target_motion_sum, pixel_count),
            "target_motion_magnitude_by_prediction_frame": target_motion_by_frame.tolist(),
            "prediction_motion_magnitude_mean": overall(prediction_motion_sum, pixel_count),
            "prediction_motion_magnitude_by_prediction_frame": prediction_motion_by_frame.tolist(),
            "motion_magnitude_bias": float(
                prediction_motion_sum.sum() / pixel_count.sum() - target_motion_sum.sum() / pixel_count.sum()
            ),
            "motion_magnitude_absolute_bias": float(
                abs(prediction_motion_sum.sum() - target_motion_sum.sum()) / pixel_count.sum()
            ),
            "motion_magnitude_mae_mean": overall(motion_magnitude_error_sum, pixel_count),
            "motion_magnitude_mae_by_prediction_frame": safe_divide(
                motion_magnitude_error_sum, pixel_count
            ),
        },
        "windows_evaluated": names,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sample_count": len(names), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
