#!/usr/bin/env python3
"""Measure the target-aware upper bound of locally routing two prediction caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"prediction", "windows"}:
            raise ValueError(f"invalid prediction cache: {path}")
        return cache["prediction"], [str(name) for name in cache["windows"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--output-cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, names = load_cache(Path(args.first_cache))
    second, second_names = load_cache(Path(args.second_cache))
    if names != second_names or first.shape != second.shape:
        raise ValueError("prediction caches are not aligned")

    horizons = first.shape[1]
    rgb_sums = {name: np.zeros(horizons, dtype=np.float64) for name in (args.first_name, args.second_name, "frame_oracle", "pixel_oracle")}
    moving_sums = {name: np.zeros(horizons, dtype=np.float64) for name in rgb_sums}
    rgb_count = np.zeros(horizons, dtype=np.float64)
    moving_count = np.zeros(horizons, dtype=np.float64)
    flow_better_pixels = np.zeros(horizons, dtype=np.float64)
    moving_flow_better_pixels = np.zeros(horizons, dtype=np.float64)
    background_flow_better_pixels = np.zeros(horizons, dtype=np.float64)
    background_count = np.zeros(horizons, dtype=np.float64)
    pixel_oracle = np.empty_like(first) if args.output_cache else None

    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            previous = np.concatenate([window["context_frames"][-1:].astype(np.float32), target[:-1]], axis=0)
        a = first[index].astype(np.float32)
        b = second[index].astype(np.float32)
        a_error = np.abs(a - target)
        b_error = np.abs(b - target)
        moving = np.abs(target - previous).mean(axis=-1) >= args.motion_threshold * 255.0
        choose_b_pixel = b_error.mean(axis=-1) < a_error.mean(axis=-1)
        choose_b_frame = b_error.mean(axis=(1, 2, 3)) < a_error.mean(axis=(1, 2, 3))
        frame_prediction = np.where(choose_b_frame[:, None, None, None], b, a)
        pixel_prediction = np.where(choose_b_pixel[..., None], b, a)
        if pixel_oracle is not None:
            pixel_oracle[index] = np.rint(pixel_prediction).clip(0, 255).astype(np.uint8)
        errors = {
            args.first_name: a_error,
            args.second_name: b_error,
            "frame_oracle": np.abs(frame_prediction - target),
            "pixel_oracle": np.abs(pixel_prediction - target),
        }
        for key, error in errors.items():
            rgb_sums[key] += error.sum(axis=(1, 2, 3), dtype=np.float64)
            moving_sums[key] += (error * moving[..., None]).sum(axis=(1, 2, 3), dtype=np.float64)
        rgb_count += np.prod(a_error.shape[1:])
        moving_count += moving.sum(axis=(1, 2), dtype=np.float64) * 3.0
        background = ~moving
        background_count += background.sum(axis=(1, 2), dtype=np.float64)
        flow_better_pixels += choose_b_pixel.sum(axis=(1, 2), dtype=np.float64)
        moving_flow_better_pixels += (choose_b_pixel & moving).sum(axis=(1, 2), dtype=np.float64)
        background_flow_better_pixels += (choose_b_pixel & background).sum(axis=(1, 2), dtype=np.float64)

    metrics = {}
    for key in rgb_sums:
        metrics[key] = {
            "rgb_mae_mean": float(rgb_sums[key].sum() / rgb_count.sum()),
            "rgb_mae_by_prediction_frame": (rgb_sums[key] / rgb_count).tolist(),
            "moving_region_rgb_mae_mean": float(moving_sums[key].sum() / moving_count.sum()),
            "moving_region_rgb_mae_by_prediction_frame": (moving_sums[key] / moving_count).tolist(),
        }
    first_mae = metrics[args.first_name]["rgb_mae_mean"]
    result = {
        "format": "track2-local-routing-oracle-v1",
        "sample_count": len(names),
        "first_name": args.first_name,
        "second_name": args.second_name,
        "motion_threshold_normalized": args.motion_threshold,
        "metrics": metrics,
        "pixel_oracle_relative_improvement_over_first": float((first_mae - metrics["pixel_oracle"]["rgb_mae_mean"]) / first_mae),
        "frame_oracle_relative_improvement_over_first": float((first_mae - metrics["frame_oracle"]["rgb_mae_mean"]) / first_mae),
        "second_better_pixel_fraction": float(flow_better_pixels.sum() / (rgb_count.sum() / 3.0)),
        "second_better_pixel_fraction_by_prediction_frame": (flow_better_pixels / (rgb_count / 3.0)).tolist(),
        "moving_region_second_better_pixel_fraction": float(moving_flow_better_pixels.sum() / (moving_count.sum() / 3.0)),
        "background_second_better_pixel_fraction": float(background_flow_better_pixels.sum() / background_count.sum()),
        "windows": names,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    if args.output_cache:
        cache_path = Path(args.output_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        with temporary_cache.open("wb") as handle:
            np.savez_compressed(handle, prediction=pixel_oracle, windows=np.asarray(names))
        os.replace(temporary_cache, cache_path)
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
