#!/usr/bin/env python3
"""Evaluate action-routed left/right AR experts against one frozen parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate_strict_track2_autoregressive_candidate import (
    Accumulator,
    Windows,
    frames,
    gains,
    load_model,
    metric_rows,
    rollout,
)


class SourceWindows(Windows):
    """Add provenance without changing the shared base dataset contract."""

    def __getitem__(self, index: int):
        values = super().__getitem__(index)
        with np.load(self.paths[index], allow_pickle=False) as data:
            is_synthetic = "synthetic_seed" in data.files
        return (*values, is_synthetic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--left-candidate", required=True)
    parser.add_argument("--right-candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset: Dataset = SourceWindows(
        Path(args.windows), split[args.episodes_key],
        min_start=args.min_start, max_start=args.max_start,
    )
    if args.max_windows and args.max_windows < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    baseline_model, baseline_mean, baseline_std = load_model(Path(args.baseline), device)
    left_model, left_mean, left_std = load_model(Path(args.left_candidate), device)
    right_model, right_mean, right_std = load_model(Path(args.right_candidate), device)
    before, after = Accumulator(), Accumulator()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for raw_context, raw_history, raw_future, raw_target, arm_right, success, is_synthetic in loader:
            context = frames(raw_context, device)
            target = frames(raw_target, device)
            raw_history, raw_future = raw_history.to(device), raw_future.to(device)
            baseline_prediction = rollout(
                baseline_model, context.clone(),
                (raw_history - baseline_mean) / baseline_std,
                (raw_future - baseline_mean) / baseline_std,
            )
            left_prediction = rollout(
                left_model, context.clone(),
                (raw_history - left_mean) / left_std,
                (raw_future - left_mean) / left_std,
            )
            right_prediction = rollout(
                right_model, context.clone(),
                (raw_history - right_mean) / right_std,
                (raw_future - right_mean) / right_std,
            )
            arm_right = arm_right.bool()
            route = arm_right.to(device)[:, None, None, None, None]
            candidate_prediction = torch.where(route, right_prediction, left_prediction)
            baseline_rows = metric_rows(baseline_prediction.float(), target, context[:, -1])
            candidate_rows = metric_rows(candidate_prediction.float(), target, context[:, -1])
            success = success.bool()
            masks = {
                "overall": torch.ones(len(arm_right), dtype=torch.bool),
                "left": ~arm_right,
                "right": arm_right,
                "capture_success": success,
                "capture_failure": ~success,
                "official": ~is_synthetic.bool(),
                "synthetic": is_synthetic.bool(),
            }
            for group, mask in masks.items():
                before.add(group, baseline_rows, mask)
                after.add(group, candidate_rows, mask)

    baseline_result, candidate_result = before.result(), after.result()
    report = {
        "format": "strict-track2-arm-routed-autoregressive-parent-evaluation-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "selected_window_count": len(dataset),
        "min_start": args.min_start,
        "max_start": args.max_start,
        "baseline_checkpoint": str(Path(args.baseline).resolve()),
        "left_candidate_checkpoint": str(Path(args.left_candidate).resolve()),
        "right_candidate_checkpoint": str(Path(args.right_candidate).resolve()),
        "routing": "raw_action_delta_argmax",
        "baseline": baseline_result,
        "candidate": candidate_result,
        "improvement": gains(baseline_result, candidate_result),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
