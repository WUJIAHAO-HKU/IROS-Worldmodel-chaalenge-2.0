#!/usr/bin/env python3
"""Cross-validate a motion gate against RGB, temporal, and texture metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from sweep_motion_gated_prediction_blend import context_motion
from sweep_prediction_blend import evenly_spaced, load_cache, motion_mask


METRIC_NAMES = (
    "rgb_mae",
    "late_rgb_mae",
    "temporal_delta_mae",
    "moving_region_rgb_mae",
    "moving_region_temporal_delta_mae",
    "moving_region_laplacian_mae",
    "motion_magnitude_mae",
)
DYNAMIC_NAMES = METRIC_NAMES[2:]


def laplacian(frames: np.ndarray) -> np.ndarray:
    padded = np.pad(frames, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return 4.0 * frames - padded[:, :, :-2, 1:-1] - padded[:, :, 2:, 1:-1] - padded[:, :, 1:-1, :-2] - padded[:, :, 1:-1, 2:]


def metric_table(prediction: np.ndarray, target: np.ndarray, context: np.ndarray, threshold: float, chunk: int = 8, workers: int = 8) -> dict[str, np.ndarray]:
    def calculate(bounds: tuple[int, int]) -> tuple[int, int, dict[str, np.ndarray]]:
        start, stop = bounds
        stop = min(start + chunk, len(target))
        pred = prediction[start:stop].astype(np.float32)
        truth = target[start:stop].astype(np.float32)
        previous_target = np.concatenate([context[start:stop, -1:], truth[:, :-1]], axis=1)
        previous_prediction = np.concatenate([context[start:stop, -1:].astype(np.float32), pred[:, :-1]], axis=1)
        target_delta = truth - previous_target
        prediction_delta = pred - previous_prediction
        rgb = np.abs(pred - truth)
        delta = np.abs(prediction_delta - target_delta)
        target_motion = np.abs(target_delta).mean(axis=-1)
        prediction_motion = np.abs(prediction_delta).mean(axis=-1)
        moving = target_motion >= threshold * 255.0
        moving_channels = moving[..., None]
        moving_count = moving.sum(axis=(1, 2, 3), dtype=np.float64) * 3.0
        count = np.maximum(moving_count, 1.0)
        values = {
            "rgb_mae": rgb.mean(axis=(1, 2, 3, 4), dtype=np.float64),
            "late_rgb_mae": rgb[:, 4:].mean(axis=(1, 2, 3, 4), dtype=np.float64),
            "temporal_delta_mae": delta.mean(axis=(1, 2, 3, 4), dtype=np.float64),
            "moving_region_rgb_mae": (rgb * moving_channels).sum(axis=(1, 2, 3, 4), dtype=np.float64) / count,
            "moving_region_temporal_delta_mae": (delta * moving_channels).sum(axis=(1, 2, 3, 4), dtype=np.float64) / count,
            "moving_region_laplacian_mae": (np.abs(laplacian(pred) - laplacian(truth)) * moving_channels).sum(axis=(1, 2, 3, 4), dtype=np.float64) / count,
            "motion_magnitude_mae": np.abs(prediction_motion - target_motion).mean(axis=(1, 2, 3), dtype=np.float64),
            "_moving_channel_count": moving_count,
        }
        return start, stop, values

    table = {name: np.empty(len(target), dtype=np.float64) for name in (*METRIC_NAMES, "_moving_channel_count")}
    bounds = [(start, min(start + chunk, len(target))) for start in range(0, len(target), chunk)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, stop, values in executor.map(calculate, bounds):
            for name, value in values.items():
                table[name][start:stop] = value
    return table


def aggregate(table: dict[str, np.ndarray], selected: np.ndarray, high_motion: np.ndarray) -> dict[str, float]:
    moving_count = table["_moving_channel_count"][selected]
    result = {}
    for name in METRIC_NAMES:
        if name.startswith("moving_region_"):
            result[name] = float((table[name][selected] * moving_count).sum() / moving_count.sum())
        else:
            result[name] = float(table[name][selected].mean())
    result["high_motion_rgb_mae"] = float(table["rgb_mae"][selected & high_motion].mean())
    return result


def combine(first: dict[str, np.ndarray], blend: dict[str, np.ndarray], use_first: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.where(use_first, first[name], blend[name]) for name in first}


def gains(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {name: (baseline[name] - candidate[name]) / baseline[name] for name in candidate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--samples", type=int, default=682)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--minimum-first-weight", type=float, default=0.5)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--maximum-appearance-regression", type=float, default=0.002)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    paths = [path for episode in split["validation_episodes"] for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))]
    paths = evenly_spaced(paths, args.samples)
    names = [path.name for path in paths]
    contexts, targets = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as window:
            contexts.append(window["context_frames"])
            targets.append(window["target_frames"])
    context, target = np.stack(contexts), np.stack(targets)
    first = load_cache(Path(args.first_cache), names, target.shape)
    second = load_cache(Path(args.second_cache), names, target.shape)
    first_table = metric_table(first, target, context, args.motion_threshold)
    high_motion = motion_mask(context, target, args.high_motion_threshold)
    tuning = np.arange(len(target)) % 2 == 0
    holdout = ~tuning
    baseline_tuning = aggregate(first_table, tuning, high_motion)
    baseline_holdout = aggregate(first_table, holdout, high_motion)
    weights = np.arange(args.minimum_first_weight, 1.0 + args.weight_step / 2.0, args.weight_step).clip(0, 1)
    weights = np.unique(np.append(weights, 1.0))
    blend_tables = []
    for weight in weights:
        if np.isclose(weight, 1.0):
            blend_tables.append(first_table)
        else:
            prediction = np.rint(first.astype(np.float32) * weight + second.astype(np.float32) * (1.0 - weight)).clip(0, 255).astype(np.uint8)
            blend_tables.append(metric_table(prediction, target, context, args.motion_threshold))
        print(json.dumps({"event": "blend_metrics", "first_weight": float(weight)}), flush=True)

    candidates = []
    for statistic in ("mean", "last"):
        observed = context_motion(context, statistic)
        thresholds = np.unique(np.concatenate(([np.nextafter(observed.min(), -np.inf)], np.quantile(observed[tuning], np.linspace(0, 1, 41)), [np.nextafter(observed.max(), np.inf)])))
        for weight_index, weight in enumerate(weights):
            for threshold in thresholds:
                use_first = observed >= threshold
                table = combine(first_table, blend_tables[weight_index], use_first)
                score = aggregate(table, tuning, high_motion)
                relative = gains(baseline_tuning, score)
                appearance_ok = all(relative[name] >= -args.maximum_appearance_regression for name in ("rgb_mae", "late_rgb_mae", "high_motion_rgb_mae"))
                dynamic_score = sum(score[name] / baseline_tuning[name] for name in DYNAMIC_NAMES) / len(DYNAMIC_NAMES)
                candidates.append({"context_motion_statistic": statistic, "context_motion_threshold": float(threshold), "low_motion_first_weight": float(weight), "appearance_ok": appearance_ok, "dynamic_relative_score": dynamic_score, "tuning": score, "tuning_relative_improvements": relative})

    eligible = [item for item in candidates if item["appearance_ok"]]
    selected = min(eligible, key=lambda item: item["dynamic_relative_score"])
    weight_index = int(np.argmin(np.abs(weights - selected["low_motion_first_weight"])))
    observed = context_motion(context, selected["context_motion_statistic"])
    use_first = observed >= selected["context_motion_threshold"]
    selected_table = combine(first_table, blend_tables[weight_index], use_first)
    selected_holdout = aggregate(selected_table, holdout, high_motion)
    holdout_relative = gains(baseline_holdout, selected_holdout)
    checks = {
        **{f"{name}_regression_within_guardrail": holdout_relative[name] >= -args.maximum_appearance_regression for name in ("rgb_mae", "late_rgb_mae", "high_motion_rgb_mae")},
        **{f"{name}_improves": holdout_relative[name] > 0.0 for name in DYNAMIC_NAMES},
    }
    selected_prediction = np.empty_like(first)
    selected_prediction[use_first] = first[use_first]
    selected_prediction[~use_first] = np.rint(first[~use_first].astype(np.float32) * selected["low_motion_first_weight"] + second[~use_first].astype(np.float32) * (1.0 - selected["low_motion_first_weight"])).clip(0, 255).astype(np.uint8)
    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, prediction=selected_prediction, windows=np.asarray(names))
    result = {"format": "track2-motion-texture-gated-blend-sweep-v1", "sample_count": len(names), "selection_protocol": "even-index tuning; odd-index holdout", "baseline_tuning": baseline_tuning, "baseline_holdout": baseline_holdout, "selected": selected, "gate_usage": {"use_first_count": int(use_first.sum()), "use_blend_count": int((~use_first).sum())}, "holdout": selected_holdout, "holdout_relative_improvements": holdout_relative, "holdout_checks": checks, "promote": all(checks.values()), "top_candidates": sorted(candidates, key=lambda item: (not item["appearance_ok"], item["dynamic_relative_score"]))[:20]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
