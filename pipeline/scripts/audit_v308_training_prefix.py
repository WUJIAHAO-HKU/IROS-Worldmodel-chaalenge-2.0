#!/usr/bin/env python3
"""Audit the currently available v304 training prefix using preregistered metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRAIN_TAGS = {
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/grad_norm",
    "train/actor/kl_beta",
    "train/actor/kl_loss",
}
ENV_TAGS = {
    "env/episode_len",
    "env/num_trajectories",
    "env/return",
    "env/reward",
    "env/success_once",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_series(
    accumulators: list[EventAccumulator], tag: str
) -> dict[int, float]:
    latest: dict[int, tuple[float, float]] = {}
    for accumulator in accumulators:
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for item in accumulator.Scalars(tag):
            candidate = (float(item.wall_time), float(item.value))
            step = int(item.step)
            if step not in latest or candidate[0] >= latest[step][0]:
                latest[step] = candidate
    return {step: latest[step][1] for step in sorted(latest)}


def main() -> int:
    args = parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    frozen = preregistration["frozen_training"]
    gates = preregistration["acceptance_gates"]
    event_files = sorted((args.run / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("no TensorBoard event files found")

    accumulators = [
        EventAccumulator(str(path), size_guidance={"scalars": 0})
        for path in event_files
    ]
    for accumulator in accumulators:
        accumulator.Reload()

    all_tags = TRAIN_TAGS | ENV_TAGS
    series = {tag: merge_series(accumulators, tag) for tag in sorted(all_tags)}
    missing_tags = sorted(tag for tag, values in series.items() if not values)
    common_train_steps = sorted(
        set.intersection(*(set(series[tag]) for tag in TRAIN_TAGS))
    ) if not any(not series[tag] for tag in TRAIN_TAGS) else []
    common_env_steps = sorted(
        set.intersection(*(set(series[tag]) for tag in ENV_TAGS))
    ) if not any(not series[tag] for tag in ENV_TAGS) else []

    train_values = {
        tag: [series[tag][step] for step in common_train_steps]
        for tag in sorted(TRAIN_TAGS)
    }
    env_values = {
        tag: [series[tag][step] for step in common_env_steps]
        for tag in sorted(ENV_TAGS)
    }
    all_values = [
        value for tag_values in (train_values | env_values).values()
        for value in tag_values
    ]
    approx_kl_limit = float(gates["action_dim_normalized_approx_kl_abs_max"])
    clip_limit = float(gates["action_dim_normalized_clip_fraction_max"])
    grad_limit = float(gates["gradient_norm_max"])
    expected_trajectories = float(frozen["trajectories_per_update"])
    expected_episode_len = float(frozen["episode_steps"])
    expected_kl_beta = float(frozen["kl_beta"])

    checks = {
        "required_tags_present": not missing_tags,
        "at_least_one_complete_train_step": bool(common_train_steps),
        "at_least_one_complete_env_step": bool(common_env_steps),
        "all_available_metrics_finite": bool(all_values)
        and all(math.isfinite(value) for value in all_values),
        "approx_kl_within_preregistered_gate": bool(common_train_steps)
        and max(abs(value) for value in train_values["train/actor/approx_kl"])
        <= approx_kl_limit,
        "clip_fraction_within_preregistered_gate": bool(common_train_steps)
        and max(train_values["train/actor/clip_fraction"]) <= clip_limit,
        "grad_norm_within_preregistered_gate": bool(common_train_steps)
        and max(train_values["train/actor/grad_norm"]) <= grad_limit,
        "kl_loss_finite_nonnegative": bool(common_train_steps)
        and all(value >= 0.0 for value in train_values["train/actor/kl_loss"]),
        "kl_beta_frozen": bool(common_train_steps)
        and all(
            math.isclose(value, expected_kl_beta, rel_tol=1e-6, abs_tol=1e-8)
            for value in train_values["train/actor/kl_beta"]
        ),
        "trajectory_budget_per_update_frozen": bool(common_env_steps)
        and all(
            math.isclose(value, expected_trajectories)
            for value in env_values["env/num_trajectories"]
        ),
        "episode_horizon_frozen": bool(common_env_steps)
        and all(
            math.isclose(value, expected_episode_len)
            for value in env_values["env/episode_len"]
        ),
        "success_once_is_probability": bool(common_env_steps)
        and all(0.0 <= value <= 1.0 for value in env_values["env/success_once"]),
    }
    report = {
        "format": "strict-track2-v304-training-prefix-audit-v1",
        "run": str(args.run),
        "preregistration": str(args.preregistration),
        "event_files": [str(path) for path in event_files],
        "common_train_steps": common_train_steps,
        "common_env_steps": common_env_steps,
        "missing_tags": missing_tags,
        "metrics": series,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": (
            "preregistered training-only metrics; no public, reserved-final, "
            "hidden, or real-submission outcomes"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
