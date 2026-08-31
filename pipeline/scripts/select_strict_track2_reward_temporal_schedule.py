#!/usr/bin/env python3
"""Select a tiny arm/output blend schedule by preregistered dynamic programming."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def split_mask(paths: list[str], bit: int) -> np.ndarray:
    return np.asarray(
        [(hashlib.sha256(Path(path).name.encode()).digest()[0] & 1) == bit for path in paths],
        dtype=bool,
    )


def arm_schedule(
    grid: list[float],
    scores: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
    mask: np.ndarray,
    smoothness: float,
    visual_incentive: float,
) -> tuple[list[float], float]:
    # scores: [strength, window, time]
    scores, target, context = scores[:, mask], target[mask], context[mask, 0]
    scale = max(float(target.std()), 1e-5)
    states = len(grid)
    absolute = np.mean(np.abs(scores - target[None]), axis=1) / scale
    cost = np.full((8, states), np.inf)
    parent = np.full((8, states), -1, dtype=np.int64)
    first_delta = np.mean(
        np.abs((scores[:, :, 0] - context[None]) - (target[:, 0] - context)[None]), axis=1
    ) / scale
    cost[0] = absolute[:, 0] + first_delta - visual_incentive * np.asarray(grid)
    for time in range(1, 8):
        for current in range(states):
            delta = np.mean(
                np.abs(
                    (scores[current, :, time][None] - scores[:, :, time - 1])
                    - (target[:, time] - target[:, time - 1])[None]
                ),
                axis=1,
            ) / scale
            transition = cost[time - 1] + delta + smoothness * np.abs(
                np.asarray(grid) - grid[current]
            )
            previous = int(np.argmin(transition))
            cost[time, current] = (
                transition[previous]
                + absolute[current, time]
                - visual_incentive * grid[current]
            )
            parent[time, current] = previous
    terminal_error = 2.0 * absolute[:, -1]
    state = int(np.argmin(cost[-1] + terminal_error))
    total = float(cost[-1, state] + terminal_error[state])
    selected = [state]
    for time in range(7, 0, -1):
        state = int(parent[time, state])
        selected.append(state)
    selected.reverse()
    return [grid[index] for index in selected], total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("STRENGTH", "REPORT"), required=True)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    expected = [float(value) for value in prereg["candidate_strength_grid"]]
    reports = {float(strength): json.loads(Path(path).read_text()) for strength, path in args.candidate}
    if sorted(reports) != sorted(expected):
        raise ValueError(f"strength grid differs from preregistration: {sorted(reports)}")
    grid = sorted(reports)
    reference = reports[grid[0]]
    paths = reference["paths"]
    target = np.asarray(reference["raw_scores"]["target"], dtype=np.float64)
    context = np.asarray(reference["raw_scores"]["context"], dtype=np.float64)
    scores = np.stack(
        [np.asarray(reports[strength]["raw_scores"]["candidate"], dtype=np.float64) for strength in grid]
    )
    with np.load(args.baseline_cache, allow_pickle=False) as values:
        right = values["arm_right"].astype(bool)
        cache_paths = values["path"].astype(str).tolist()
    if paths != cache_paths:
        raise ValueError("reward reports and baseline cache are not aligned")
    selection = split_mask(paths, 0)
    confirmation = split_mask(paths, 1)
    candidates = []
    cfg = prereg["dynamic_program_cost"]
    for incentive in cfg["visual_strength_incentives"]:
        left, left_cost = arm_schedule(
            grid, scores, target, context, selection & ~right,
            float(cfg["strength_smoothness_weight"]), float(incentive),
        )
        right_schedule, right_cost = arm_schedule(
            grid, scores, target, context, selection & right,
            float(cfg["strength_smoothness_weight"]), float(incentive),
        )
        candidates.append(
            {
                "name": f"visual_{str(incentive).replace('.', 'p')}",
                "visual_strength_incentive": incentive,
                "left_schedule": left,
                "right_schedule": right_schedule,
                "selection_cost": left_cost + right_cost,
            }
        )
    result = {
        "format": "strict-track2-reward-temporal-schedule-selection-v1",
        "preregistration": str(args.preregistration.resolve()),
        "selection_windows": int(selection.sum()),
        "confirmation_windows": int(confirmation.sum()),
        "selection_left": int((selection & ~right).sum()),
        "selection_right": int((selection & right).sum()),
        "confirmation_left": int((confirmation & ~right).sum()),
        "confirmation_right": int((confirmation & right).sum()),
        "grid": grid,
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
