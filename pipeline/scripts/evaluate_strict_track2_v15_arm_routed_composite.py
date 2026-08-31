#!/usr/bin/env python3
"""Paired evaluation of frozen V15 and a direct arm-routed V15 adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import Accumulator, Windows, gains, metric_rows
from wam_pipeline.v15_arm_routed_runtime import Track2V15ArmRoutedRuntime
from wam_pipeline.v15_runtime import Track2V15Runtime


def tensor_frames(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline-release", required=True)
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = Windows(
        Path(args.windows), split[args.episodes_key],
        min_start=args.min_start, max_start=args.max_start,
    )
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(indices) - 1, args.max_windows, dtype=int).tolist()
    baseline = Track2V15Runtime(args.baseline_release, args.library, args.device)
    candidate = Track2V15ArmRoutedRuntime(args.candidate_release, args.library, args.device)
    before, after = Accumulator(), Accumulator()
    routes, selected_paths = [], []
    for position, index in enumerate(indices, 1):
        context, history, future, target, arm_right, success = dataset[index]
        baseline_prediction = baseline.predict(context, history, future, 0, None)
        candidate_prediction = candidate.predict(context, history, future, 0, None)
        context_tensor = tensor_frames(context)[0]
        target_tensor = tensor_frames(target)
        baseline_rows = metric_rows(
            tensor_frames(baseline_prediction), target_tensor, context_tensor[-1:]
        )
        candidate_rows = metric_rows(
            tensor_frames(candidate_prediction), target_tensor, context_tensor[-1:]
        )
        arm_mask = torch.tensor([bool(arm_right)])
        success_mask = torch.tensor([bool(success)])
        with np.load(dataset.paths[index], allow_pickle=False) as values:
            synthetic_mask = torch.tensor(["synthetic_seed" in values.files])
        masks = {
            "overall": torch.ones(1, dtype=torch.bool),
            "left": ~arm_mask,
            "right": arm_mask,
            "capture_success": success_mask,
            "capture_failure": ~success_mask,
            "official": ~synthetic_mask,
            "synthetic": synthetic_mask,
        }
        for group, mask in masks.items():
            before.add(group, baseline_rows, mask)
            after.add(group, candidate_rows, mask)
        route = candidate.arm_routed_autoregressive.last_route
        routes.append({
            "path": dataset.paths[index].name,
            "route": route,
            "bit_exact_to_baseline": bool(np.array_equal(baseline_prediction, candidate_prediction)),
        })
        selected_paths.append(dataset.paths[index].name)
        if position == 1 or position % 4 == 0 or position == len(indices):
            print(json.dumps({"evaluated": position, "total": len(indices), "route": route}), flush=True)

    baseline_result, candidate_result = before.result(), after.result()
    report = {
        "format": "strict-track2-v15-direct-arm-routed-composite-evaluation-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "min_start": args.min_start,
        "max_start": args.max_start,
        "selected_window_count": len(indices),
        "selected_paths": selected_paths,
        "baseline_release": str(Path(args.baseline_release).resolve()),
        "candidate_release": str(Path(args.candidate_release).resolve()),
        "routing": {
            "candidate_left": sum(row["route"] == "candidate_left" for row in routes),
            "candidate_right": sum(row["route"] == "candidate_right" for row in routes),
            "bit_exact_to_baseline": sum(row["bit_exact_to_baseline"] for row in routes),
            "records": routes,
        },
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
