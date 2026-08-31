#!/usr/bin/env python3
"""Cross-validate inference-only motion gain and moving-region sharpening."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_grid(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",")})
    if not result:
        raise ValueError("empty calibration grid")
    return result


def box_blur(frames: np.ndarray) -> np.ndarray:
    padded = np.pad(frames, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(padded[:, :, y : y + frames.shape[2], x : x + frames.shape[3]] for y in range(3) for x in range(3)) / 9.0


def laplacian(frames: np.ndarray) -> np.ndarray:
    padded = np.pad(frames, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return 4.0 * frames - padded[:, :, :-2, 1:-1] - padded[:, :, 2:, 1:-1] - padded[:, :, 1:-1, :-2] - padded[:, :, 1:-1, 2:]


def calibrate(prediction: np.ndarray, context_last: np.ndarray, gain: float, sharpen: float, gate_scale: float) -> np.ndarray:
    raw = prediction.astype(np.float32)
    reference_previous = context_last.astype(np.float32)
    output_previous = reference_previous.copy()
    motion_adjusted = np.empty_like(raw)
    gates = np.empty(raw.shape[:-1] + (1,), dtype=np.float32)
    for horizon in range(raw.shape[1]):
        raw_delta = raw[:, horizon] - reference_previous[:, 0]
        gate = np.clip(np.abs(raw_delta).mean(axis=-1, keepdims=True) / gate_scale, 0.0, 1.0)
        output = output_previous[:, 0] + raw_delta * (1.0 + (gain - 1.0) * gate)
        motion_adjusted[:, horizon] = output
        gates[:, horizon] = gate
        reference_previous = raw[:, horizon : horizon + 1]
        output_previous = output[:, None]
    if sharpen:
        motion_adjusted += sharpen * gates * (motion_adjusted - box_blur(motion_adjusted))
    return np.rint(motion_adjusted).clip(0, 255).astype(np.uint8)


def calculate_metrics(prediction: np.ndarray, target: np.ndarray, context_last: np.ndarray, selected: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prediction[selected].astype(np.float32)
    truth = target[selected].astype(np.float32)
    context = context_last[selected].astype(np.float32)
    target_previous = np.concatenate([context, truth[:, :-1]], axis=1)
    prediction_previous = np.concatenate([context, pred[:, :-1]], axis=1)
    target_delta = truth - target_previous
    prediction_delta = pred - prediction_previous
    rgb_error = np.abs(pred - truth)
    delta_error = np.abs(prediction_delta - target_delta)
    target_motion = np.abs(target_delta).mean(axis=-1)
    prediction_motion = np.abs(prediction_delta).mean(axis=-1)
    moving = target_motion >= threshold * 255.0
    moving_count = float(moving.sum() * 3)
    moving_channels = moving[..., None]
    high_motion = target_motion.mean(axis=(1, 2, 3)) / 255.0 >= 0.04
    return {
        "rgb_mae": float(rgb_error.mean()),
        "late_rgb_mae": float(rgb_error[:, 4:].mean()),
        "high_motion_rgb_mae": float(rgb_error[high_motion].mean()) if high_motion.any() else float("nan"),
        "temporal_delta_mae": float(delta_error.mean()),
        "moving_region_rgb_mae": float((rgb_error * moving_channels).sum() / moving_count),
        "moving_region_temporal_delta_mae": float((delta_error * moving_channels).sum() / moving_count),
        "moving_region_laplacian_mae": float((np.abs(laplacian(pred) - laplacian(truth)) * moving_channels).sum() / moving_count),
        "motion_magnitude_mae": float(np.abs(prediction_motion - target_motion).mean()),
        "prediction_motion_magnitude": float(prediction_motion.mean()),
    }


def relative_gain(baseline: dict[str, float], candidate: dict[str, float], name: str) -> float:
    return (baseline[name] - candidate[name]) / baseline[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--motion-gains", default="1.0,1.025,1.05,1.075,1.1,1.125,1.15")
    parser.add_argument("--sharpen-weights", default="0.0,0.025,0.05,0.075,0.1,0.125,0.15")
    parser.add_argument("--prediction-motion-gate", type=float, default=0.03)
    parser.add_argument("--target-motion-threshold", type=float, default=0.03)
    args = parser.parse_args()
    gains, sharpens = parse_grid(args.motion_gains), parse_grid(args.sharpen_weights)
    if min(gains) < 1.0 or min(sharpens) < 0.0 or args.prediction_motion_gate <= 0.0:
        raise SystemExit("motion gains must be >= 1, sharpening >= 0, and gate > 0")

    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        prediction = cache["prediction"]
        names = [str(name) for name in cache["windows"]]
    targets, contexts = [], []
    for name in names:
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            targets.append(window["target_frames"])
            contexts.append(window["context_frames"][-1:])
    target = np.stack(targets)
    context_last = np.stack(contexts)
    tuning = np.arange(len(names)) % 2 == 0
    holdout = ~tuning
    baseline_tuning = calculate_metrics(prediction, target, context_last, tuning, args.target_motion_threshold)
    baseline_holdout = calculate_metrics(prediction, target, context_last, holdout, args.target_motion_threshold)
    dynamic_names = (
        "temporal_delta_mae",
        "moving_region_temporal_delta_mae",
        "moving_region_laplacian_mae",
        "motion_magnitude_mae",
    )
    candidates = []
    best_key = None
    best_score = float("inf")
    for gain in gains:
        for sharpen in sharpens:
            calibrated = calibrate(prediction, context_last, gain, sharpen, args.prediction_motion_gate * 255.0)
            metrics = calculate_metrics(calibrated, target, context_last, tuning, args.target_motion_threshold)
            relative = {name: relative_gain(baseline_tuning, metrics, name) for name in metrics}
            appearance_ok = all(relative[name] >= -0.002 for name in ("rgb_mae", "late_rgb_mae", "high_motion_rgb_mae"))
            score = sum(metrics[name] / baseline_tuning[name] for name in dynamic_names) / len(dynamic_names)
            candidates.append({"motion_gain": gain, "sharpen_weight": sharpen, "appearance_ok": appearance_ok, "dynamic_relative_score": score, "tuning": metrics, "tuning_relative_improvements": relative})
            if appearance_ok and score < best_score:
                best_key, best_score = (gain, sharpen), score
    if best_key is None:
        raise RuntimeError("no calibration candidate satisfies the appearance guardrail")
    selected_prediction = calibrate(prediction, context_last, *best_key, args.prediction_motion_gate * 255.0)
    selected_tuning = calculate_metrics(selected_prediction, target, context_last, tuning, args.target_motion_threshold)
    selected_holdout = calculate_metrics(selected_prediction, target, context_last, holdout, args.target_motion_threshold)
    holdout_gains = {name: relative_gain(baseline_holdout, selected_holdout, name) for name in selected_holdout}
    holdout_checks = {
        "rgb_regression_within_0.2_percent": holdout_gains["rgb_mae"] >= -0.002,
        "late_rgb_regression_within_0.2_percent": holdout_gains["late_rgb_mae"] >= -0.002,
        "high_motion_rgb_regression_within_0.2_percent": holdout_gains["high_motion_rgb_mae"] >= -0.002,
        **{f"{name}_does_not_regress": holdout_gains[name] >= 0.0 for name in dynamic_names},
        "at_least_one_dynamic_metric_improves_0.25_percent": max(
            holdout_gains[name] for name in dynamic_names
        ) >= 0.0025,
    }
    result = {
        "format": "track2-motion-texture-calibration-sweep-v1",
        "prediction_cache": str(Path(args.prediction_cache).resolve()),
        "sample_count": len(names),
        "selection_protocol": "even-index tuning; odd-index holdout",
        "baseline_tuning": baseline_tuning,
        "baseline_holdout": baseline_holdout,
        "selected": {"motion_gain": best_key[0], "sharpen_weight": best_key[1], "prediction_motion_gate": args.prediction_motion_gate, "tuning": selected_tuning, "holdout": selected_holdout, "holdout_relative_improvements": holdout_gains, "holdout_checks": holdout_checks, "promote": all(holdout_checks.values())},
        "candidates": sorted(candidates, key=lambda item: (not item["appearance_ok"], item["dynamic_relative_score"])),
    }
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, prediction=selected_prediction, windows=np.asarray(names))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
