#!/usr/bin/env python3
"""Compute the standard paired visual metrics from cached rollout tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_track2_autoregressive_candidate import Accumulator, gains, metric_rows


def tensor_frames(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)


def tensor_frame(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reuse-baseline-cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as values:
        candidates = values["candidate"].copy()
        candidate_paths = values["path"].astype(str).tolist()
    source = args.reuse_baseline_cache or args.cache
    with np.load(source, allow_pickle=False) as values:
        context = values["context_last"].copy()
        target = values["target"].copy()
        baseline = values["baseline"].copy()
        arm_right = values["arm_right"].astype(bool)
        capture_success = values["capture_success"].astype(bool)
        paths = values["path"].astype(str).tolist()
    if paths != candidate_paths:
        raise ValueError("candidate and baseline caches have different paths")
    before, after = Accumulator(), Accumulator()
    for index in range(len(paths)):
        target_tensor = tensor_frames(target[index])
        base_rows = metric_rows(
            tensor_frames(baseline[index]), target_tensor, tensor_frame(context[index])
        )
        candidate_rows = metric_rows(
            tensor_frames(candidates[index]), target_tensor, tensor_frame(context[index])
        )
        masks = {
            "overall": torch.ones(1, dtype=torch.bool),
            "left": torch.tensor([not arm_right[index]]),
            "right": torch.tensor([arm_right[index]]),
            "capture_success": torch.tensor([capture_success[index]]),
            "capture_failure": torch.tensor([not capture_success[index]]),
        }
        for group, mask in masks.items():
            before.add(group, base_rows, mask)
            after.add(group, candidate_rows, mask)
    baseline_result, candidate_result = before.result(), after.result()
    report = {
        "format": "strict-track2-cached-candidate-visual-evaluation-v1",
        "cache": str(Path(args.cache).resolve()),
        "reuse_baseline_cache": str(Path(source).resolve()),
        "window_count": len(paths),
        "baseline": baseline_result,
        "candidate": candidate_result,
        "improvement": gains(baseline_result, candidate_result),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["improvement"], indent=2))


if __name__ == "__main__":
    main()
