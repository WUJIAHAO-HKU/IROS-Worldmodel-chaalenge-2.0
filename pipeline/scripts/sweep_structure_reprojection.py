#!/usr/bin/env python3
"""Cross-validate conservative high-frequency injection from a reprojection parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def blur(value: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    shape = value.shape[:2]
    return functional.avg_pool2d(value.flatten(0, 1), kernel, stride=1, padding=kernel // 2, count_include_pad=False).unflatten(0, shape)


def add_metrics(sums: dict[str, float], value: torch.Tensor, target: torch.Tensor, context: torch.Tensor) -> None:
    error = (value - target).abs()
    high_error = ((value - blur(value)) - (target - blur(target))).abs()
    edge_x = ((value[..., 1:] - value[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    edge_y = ((value[..., 1:, :] - value[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    previous = torch.cat((context, target[:, :-1]), dim=1)
    moving = (target - previous).abs().mean(dim=2, keepdim=True) >= 0.03
    moving_channels = moving.expand_as(target)
    dark = target.mean(dim=2, keepdim=True) < 0.30
    dark_channels = dark.expand_as(target)
    sums["rgb_sum"] += float(error.sum())
    sums["rgb_count"] += error.numel()
    sums["high_sum"] += float(high_error.sum())
    sums["high_count"] += high_error.numel()
    sums["edge_sum"] += float(edge_x.sum() + edge_y.sum())
    sums["edge_count"] += edge_x.numel() + edge_y.numel()
    sums["dark_sum"] += float(error[dark_channels].sum())
    sums["dark_count"] += int(dark_channels.sum())
    sums["moving_rgb_sum"] += float(error[moving_channels].sum())
    sums["moving_high_sum"] += float(high_error[moving_channels].sum())
    sums["moving_count"] += int(moving_channels.sum())


def finish(sums: dict[str, float]) -> dict[str, float]:
    return {
        "rgb_mae": sums["rgb_sum"] / sums["rgb_count"],
        "highpass_mae": sums["high_sum"] / sums["high_count"],
        "edge_mae": sums["edge_sum"] / sums["edge_count"],
        "dark_region_rgb_mae": sums["dark_sum"] / sums["dark_count"],
        "moving_region_rgb_mae": sums["moving_rgb_sum"] / sums["moving_count"],
        "moving_region_highpass_mae": sums["moving_high_sum"] / sums["moving_count"],
    }


def empty() -> dict[str, float]:
    return {name: 0.0 for name in ("rgb_sum", "rgb_count", "high_sum", "high_count", "edge_sum", "edge_count", "dark_sum", "dark_count", "moving_rgb_sum", "moving_high_sum", "moving_count")}


def candidate(first: torch.Tensor, second: torch.Tensor, context: torch.Tensor, gain: float, disagreement_threshold: float, edge_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    first_low, second_low = blur(first), blur(second)
    first_high, second_high = first - first_low, second - second_low
    disagreement = (first_low - second_low).abs().mean(dim=2, keepdim=True)
    geometry = (1.0 - disagreement / disagreement_threshold).clamp(0.0, 1.0)
    edge_advantage = second_high.abs().mean(dim=2, keepdim=True) - first_high.abs().mean(dim=2, keepdim=True)
    structure = (edge_advantage / edge_scale).clamp(0.0, 1.0)
    motion = (first - context).abs().mean(dim=2, keepdim=True)
    motion_support = 0.25 + 0.75 * (motion / 0.03).clamp(0.0, 1.0)
    confidence = geometry * structure * motion_support
    value = first + gain * confidence * (second_high - first_high)
    return value.clamp(0.0, 1.0), confidence


def score(baseline: dict[str, float], value: dict[str, float]) -> float:
    weights = {"rgb_mae": 0.35, "highpass_mae": 0.25, "edge_mae": 0.15, "dark_region_rgb_mae": 0.10, "moving_region_rgb_mae": 0.10, "moving_region_highpass_mae": 0.05}
    result = sum(weight * value[name] / baseline[name] for name, weight in weights.items())
    result += 20.0 * max(value["rgb_mae"] / baseline["rgb_mae"] - 1.0005, 0.0)
    result += 5.0 * max(value["dark_region_rgb_mae"] / baseline["dark_region_rgb_mae"] - 1.001, 0.0)
    return float(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--reprojection-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.first_cache, allow_pickle=False) as cache:
        first = cache["prediction"]
        names = [str(value) for value in cache["windows"]]
    with np.load(args.reprojection_cache, allow_pickle=False) as cache:
        second = cache["prediction"]
        second_names = [str(value) for value in cache["windows"]]
    if names != second_names or first.shape != second.shape:
        raise ValueError("prediction caches are not aligned")
    configs = [
        {"gain": gain, "disagreement_threshold": threshold, "edge_scale": scale}
        for gain in (0.25, 0.5, 0.75, 1.0)
        for threshold in (0.02, 0.04, 0.06)
        for scale in (0.005, 0.01)
    ]
    labels = ("selection_even", "blind_odd")
    baseline_sums = {label: empty() for label in labels}
    candidate_sums = [{label: empty() for label in labels} for _ in configs]
    confidence_sums = [0.0] * len(configs)
    confidence_counts = [0] * len(configs)
    device = torch.device(args.device)
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target_numpy = window["target_frames"]
            context_numpy = window["context_frames"][-1:]
        first_batch = torch.from_numpy(first[index:index + 1].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        second_batch = torch.from_numpy(second[index:index + 1].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        target = torch.from_numpy(target_numpy[None].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        context = torch.from_numpy(context_numpy[None].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        label = labels[index % 2]
        add_metrics(baseline_sums[label], first_batch, target, context)
        for config_index, config in enumerate(configs):
            value, confidence = candidate(first_batch, second_batch, context, **config)
            add_metrics(candidate_sums[config_index][label], value, target, context)
            confidence_sums[config_index] += float(confidence.sum())
            confidence_counts[config_index] += confidence.numel()
        if (index + 1) % 100 == 0 or index + 1 == len(names):
            print(json.dumps({"completed": index + 1, "total": len(names)}), flush=True)
    baseline = {label: finish(value) for label, value in baseline_sums.items()}
    records = []
    for config, values, confidence_sum, confidence_count in zip(configs, candidate_sums, confidence_sums, confidence_counts):
        metrics = {label: finish(values[label]) for label in labels}
        records.append({"config": config, "confidence_mean": confidence_sum / confidence_count, "selection_score": score(baseline["selection_even"], metrics["selection_even"]), "blind_score": score(baseline["blind_odd"], metrics["blind_odd"]), "metrics": metrics})
    best = min(records, key=lambda value: value["selection_score"])
    result = {"format": "track2-structure-reprojection-sweep-v1", "sample_count": len(names), "first_cache": str(Path(args.first_cache).resolve()), "reprojection_cache": str(Path(args.reprojection_cache).resolve()), "baseline": baseline, "best": best, "candidates": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "baseline": baseline, "best": best}, indent=2))


if __name__ == "__main__":
    main()
