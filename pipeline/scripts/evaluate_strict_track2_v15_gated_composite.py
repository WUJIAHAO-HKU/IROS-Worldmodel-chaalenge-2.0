#!/usr/bin/env python3
"""Paired evaluation of the complete frozen and gated V15 composites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import (
    Accumulator,
    Windows,
    gains,
    metric_rows,
)
from wam_pipeline.v15_gated_runtime import Track2V15GatedRuntime
from wam_pipeline.v15_runtime import Track2V15Runtime


def tensor_frames(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline-release", required=True)
    parser.add_argument("--gated-release", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output-strengths", default="1,1.5,2,2.5,3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset = Windows(Path(args.windows), split[args.episodes_key])
    indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(indices):
        indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()
    baseline = Track2V15Runtime(args.baseline_release, args.library, args.device)
    gated = Track2V15GatedRuntime(args.gated_release, args.library, args.device)
    strengths = [float(value) for value in args.output_strengths.split(",")]
    if not strengths or any(value <= 0 for value in strengths):
        raise ValueError("output strengths must be positive")
    before = Accumulator()
    after = {strength: Accumulator() for strength in strengths}
    routes, selected_paths = [], []
    for position, index in enumerate(indices, 1):
        context, history, future, target, arm_right, success = dataset[index]
        baseline_prediction = baseline.predict(context, history, future, 0, None)
        gated_prediction = gated.predict(context, history, future, 0, None)
        context_tensor = tensor_frames(context)[0]
        target_tensor = tensor_frames(target)
        baseline_rows = metric_rows(tensor_frames(baseline_prediction), target_tensor, context_tensor[-1:])
        candidate_rows = {}
        baseline_float = baseline_prediction.astype(np.float32)
        gated_delta = gated_prediction.astype(np.float32) - baseline_float
        for strength in strengths:
            candidate = np.clip(np.rint(baseline_float + strength * gated_delta), 0, 255).astype(np.uint8)
            candidate_rows[strength] = metric_rows(
                tensor_frames(candidate), target_tensor, context_tensor[-1:]
            )
        arm_mask = torch.tensor([bool(arm_right)])
        success_mask = torch.tensor([bool(success)])
        masks = {
            "overall": torch.ones(1, dtype=torch.bool),
            "left": ~arm_mask,
            "right": arm_mask,
            "capture_success": success_mask,
            "capture_failure": ~success_mask,
        }
        for group, mask in masks.items():
            before.add(group, baseline_rows, mask)
            for strength in strengths:
                after[strength].add(group, candidate_rows[strength], mask)
        routes.append({
            "path": dataset.paths[index].name,
            "probability_synthetic": gated.gated_autoregressive.last_probability_synthetic,
            "route": gated.gated_autoregressive.last_route,
            "bit_exact_to_baseline": bool(np.array_equal(baseline_prediction, gated_prediction)),
        })
        selected_paths.append(dataset.paths[index].name)
        if position == 1 or position % 4 == 0 or position == len(indices):
            print(json.dumps({"evaluated": position, "total": len(indices), "last_route": routes[-1]}), flush=True)

    baseline_result = before.result()
    candidate_by_strength = {
        str(strength): accumulator.result() for strength, accumulator in after.items()
    }
    improvement_by_strength = {
        str(strength): gains(baseline_result, candidate_by_strength[str(strength)])
        for strength in strengths
    }
    eligible = []
    for strength in strengths:
        record = improvement_by_strength[str(strength)]["overall"]
        if (
            record["texture_mae_improvement_percent"] >= -.5
            and record["temporal_delta_mae_improvement_percent"] >= -.5
        ):
            eligible.append((record["rgb_mae_improvement_percent"], -strength, strength))
    selected_strength = max(eligible)[2] if eligible else 1.0
    gated_result = candidate_by_strength[str(selected_strength)]
    report = {
        "format": "strict-track2-v15-gated-composite-paired-evaluation-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "selected_window_count": len(indices),
        "selected_paths": selected_paths,
        "baseline_release": str(Path(args.baseline_release).resolve()),
        "gated_release": str(Path(args.gated_release).resolve()),
        "routing": {
            "candidate": sum(
                item["route"].startswith("candidate") for item in routes
            ),
            "candidate_left": sum(
                item["route"] == "candidate_left" for item in routes
            ),
            "candidate_right": sum(
                item["route"] == "candidate_right" for item in routes
            ),
            "baseline": sum(item["route"] == "baseline" for item in routes),
            "bit_exact_to_baseline": sum(item["bit_exact_to_baseline"] for item in routes),
            "records": routes,
        },
        "output_strengths": strengths,
        "selected_output_strength": selected_strength,
        "baseline": baseline_result,
        "candidate": gated_result,
        "improvement": gains(baseline_result, gated_result),
        "candidate_by_output_strength": candidate_by_strength,
        "improvement_by_output_strength": improvement_by_strength,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
