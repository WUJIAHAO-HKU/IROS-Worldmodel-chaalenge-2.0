#!/usr/bin/env python3
"""Evaluate a learned state router between frozen and adapted Track 2 parents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def load_gate(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "strict-track2-visual-source-gate-v1":
        raise ValueError("unsupported source gate")
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def gate_probability(context: torch.Tensor, gate: dict) -> torch.Tensor:
    last = context[:, -1]
    pooled = F.adaptive_avg_pool2d(last, (int(gate["pool_size"]), int(gate["pool_size"]))).flatten(1)
    feature = torch.cat((pooled, last.mean((2, 3)), last.std((2, 3), unbiased=False)), 1)
    normalized = (feature - gate["feature_mean"]) / gate["feature_std"]
    return torch.sigmoid(F.linear(normalized, gate["weight"], gate["bias"]))


def gated_rollout(
    baseline_model,
    candidate_model,
    context,
    raw_history,
    raw_future,
    baseline_mean,
    baseline_std,
    candidate_mean,
    candidate_std,
    route,
):
    baseline_history = (raw_history - baseline_mean) / baseline_std
    candidate_history = (raw_history - candidate_mean) / candidate_std
    baseline_future = (raw_future - baseline_mean) / baseline_std
    candidate_future = (raw_future - candidate_mean) / candidate_std
    output = []
    for baseline_action, candidate_action in zip(baseline_future.unbind(1), candidate_future.unbind(1)):
        baseline_value = baseline_model(
            context, torch.cat((baseline_history, baseline_action[:, None]), 1)
        ).clamp(0, 1)
        candidate_value = candidate_model(
            context, torch.cat((candidate_history, candidate_action[:, None]), 1)
        ).clamp(0, 1)
        value = baseline_value + route[:, :, None, None] * (candidate_value - baseline_value)
        output.append(value)
        context = torch.cat((context[:, 1:], value[:, None]), 1)
        baseline_history = torch.cat((baseline_history[:, 1:], baseline_action[:, None]), 1)
        candidate_history = torch.cat((candidate_history[:, 1:], candidate_action[:, None]), 1)
    return torch.stack(output, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--soft-gate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset: Dataset = Windows(Path(args.windows), split[args.episodes_key])
    if args.max_windows and args.max_windows < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    baseline_model, baseline_mean, baseline_std = load_model(Path(args.baseline), device)
    candidate_model, candidate_mean, candidate_std = load_model(Path(args.candidate), device)
    gate = load_gate(Path(args.source_gate), device)
    before, after = Accumulator(), Accumulator()
    route_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for raw_context, raw_history, raw_future, raw_target, arm_right, success in loader:
            context = frames(raw_context, device)
            target = frames(raw_target, device)
            raw_history, raw_future = raw_history.to(device), raw_future.to(device)
            probability = gate_probability(context.float(), gate)
            route = probability if args.soft_gate else (probability >= float(gate["threshold"])).to(probability.dtype)
            baseline_prediction = rollout(
                baseline_model,
                context.clone(),
                (raw_history - baseline_mean) / baseline_std,
                (raw_future - baseline_mean) / baseline_std,
            )
            candidate_prediction = gated_rollout(
                baseline_model, candidate_model, context.clone(), raw_history, raw_future,
                baseline_mean, baseline_std, candidate_mean, candidate_std, route,
            )
            baseline_rows = metric_rows(baseline_prediction.float(), target, context[:, -1])
            candidate_rows = metric_rows(candidate_prediction.float(), target, context[:, -1])
            arm_right, success = arm_right.bool(), success.bool()
            masks = {
                "overall": torch.ones(len(arm_right), dtype=torch.bool),
                "left": ~arm_right,
                "right": arm_right,
                "capture_success": success,
                "capture_failure": ~success,
            }
            for group, mask in masks.items():
                before.add(group, baseline_rows, mask)
                after.add(group, candidate_rows, mask)
                if bool(mask.any()):
                    route_values[group].append(probability[mask.to(probability.device)].cpu())

    baseline_result, candidate_result = before.result(), after.result()
    routing = {}
    for group, chunks in route_values.items():
        values = torch.cat(chunks).flatten()
        routing[group] = {
            "windows": len(values),
            "mean_probability_synthetic": float(values.mean()),
            "routed_to_candidate": int((values >= float(gate["threshold"])).sum()),
        }
    report = {
        "format": "strict-track2-gated-autoregressive-parent-evaluation-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "selected_window_count": len(dataset),
        "baseline_checkpoint": str(Path(args.baseline).resolve()),
        "candidate_checkpoint": str(Path(args.candidate).resolve()),
        "source_gate": str(Path(args.source_gate).resolve()),
        "routing_mode": "soft" if args.soft_gate else "hard_at_0_5",
        "routing": routing,
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
