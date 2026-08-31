#!/usr/bin/env python3
"""Compare edge, dark-region, and structure-mask metrics for aligned caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def highpass(value):
    flat = value.flatten(0, 1)
    blurred = functional.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
    return (flat - blurred).unflatten(0, value.shape[:2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.baseline_cache, allow_pickle=False) as cache:
        baseline, names = cache["prediction"], [str(value) for value in cache["windows"]]
    with np.load(args.candidate_cache, allow_pickle=False) as cache:
        candidate, candidate_names = cache["prediction"], [str(value) for value in cache["windows"]]
    with np.load(args.mask_cache, allow_pickle=False) as cache:
        masks, mask_names = cache["structure_mask"], [str(value) for value in cache["windows"]]
    if names != candidate_names or names != mask_names or baseline.shape != candidate.shape or masks.shape != baseline.shape[:2] + baseline.shape[2:4]:
        raise ValueError("caches are not aligned")
    device = torch.device(args.device)
    metric_names = ("rgb", "highpass", "edge", "dark_rgb", "mask_rgb", "mask_highpass", "mask_edge", "texture_roi_rgb", "texture_roi_highpass")
    sums = {model: {metric: 0.0 for metric in metric_names} for model in ("baseline", "candidate")}
    counts = {metric: 0.0 for metric in metric_names}
    for start in range(0, len(names), args.batch_size):
        stop = min(start + args.batch_size, len(names))
        targets = []
        for name in names[start:stop]:
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                targets.append(window["target_frames"])
        target = torch.from_numpy(np.stack(targets)).permute(0, 1, 4, 2, 3).to(device).float().div(255.0)
        values = {
            "baseline": torch.from_numpy(baseline[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0),
            "candidate": torch.from_numpy(candidate[start:stop].copy()).permute(0, 1, 4, 2, 3).to(device).float().div(255.0),
        }
        mask = torch.from_numpy(masks[start:stop, :, None].copy()).to(device).float().div(255.0)
        mask_channels = mask.expand_as(target)
        target_hp = highpass(target)
        dark = (target.mean(dim=2, keepdim=True) < 0.30).expand_as(target)
        texture_roi = (target_hp.abs().mean(dim=2, keepdim=True) >= 0.03).expand_as(target)
        mask_edge_x = 0.5 * (mask[..., 1:] + mask[..., :-1])
        mask_edge_y = 0.5 * (mask[..., 1:, :] + mask[..., :-1, :])
        for model_name, value in values.items():
            error = (value - target).abs()
            hp_error = (highpass(value) - target_hp).abs()
            edge_x = ((value[..., 1:] - value[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
            edge_y = ((value[..., 1:, :] - value[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
            sums[model_name]["rgb"] += float(error.sum())
            sums[model_name]["highpass"] += float(hp_error.sum())
            sums[model_name]["edge"] += float(edge_x.sum() + edge_y.sum())
            sums[model_name]["dark_rgb"] += float(error[dark].sum())
            sums[model_name]["mask_rgb"] += float((mask_channels * error).sum())
            sums[model_name]["mask_highpass"] += float((mask_channels * hp_error).sum())
            sums[model_name]["mask_edge"] += float((mask_edge_x * edge_x).sum() + (mask_edge_y * edge_y).sum())
            sums[model_name]["texture_roi_rgb"] += float(error[texture_roi].sum())
            sums[model_name]["texture_roi_highpass"] += float(hp_error[texture_roi].sum())
        counts["rgb"] += target.numel()
        counts["highpass"] += target.numel()
        counts["edge"] += edge_x.numel() + edge_y.numel()
        counts["dark_rgb"] += int(dark.sum())
        counts["mask_rgb"] += float(mask_channels.sum())
        counts["mask_highpass"] += float(mask_channels.sum())
        counts["mask_edge"] += float(mask_edge_x.sum() * 3.0 + mask_edge_y.sum() * 3.0)
        counts["texture_roi_rgb"] += int(texture_roi.sum())
        counts["texture_roi_highpass"] += int(texture_roi.sum())
        if stop % 100 == 0 or stop == len(names):
            print(json.dumps({"completed": stop, "total": len(names)}), flush=True)
    metrics = {model: {name: 255.0 * sums[model][name] / max(counts[name], 1.0) for name in metric_names} for model in sums}
    relative = {name: 100.0 * (metrics["baseline"][name] - metrics["candidate"][name]) / metrics["baseline"][name] for name in metric_names}
    result = {"format": "track2-prediction-structure-comparison-v1", "sample_count": len(names), "baseline_cache": str(Path(args.baseline_cache).resolve()), "candidate_cache": str(Path(args.candidate_cache).resolve()), "mask_cache": str(Path(args.mask_cache).resolve()), "units": "0-255 MAE", "baseline": metrics["baseline"], "candidate": metrics["candidate"], "relative_improvement_percent": relative}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
