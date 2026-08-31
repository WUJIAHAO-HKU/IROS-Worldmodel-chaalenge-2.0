#!/usr/bin/env python3
"""Cross-validate an observable-motion gate for two Track 2 prediction caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sweep_prediction_blend import blended_error, evenly_spaced, load_cache, metrics, motion_mask, prediction_error


def context_motion(context: np.ndarray, statistic: str) -> np.ndarray:
    difference = np.abs(context[:, 1:].astype(np.int16) - context[:, :-1].astype(np.int16))
    per_transition = difference.mean(axis=(2, 3, 4), dtype=np.float64) / 255.0
    if statistic == "last":
        return per_transition[:, -1]
    if statistic == "mean":
        return per_transition.mean(axis=1)
    raise ValueError(f"unsupported context-motion statistic: {statistic}")


def gated_error(
    first_error: np.ndarray,
    blended: np.ndarray,
    observed_motion: np.ndarray,
    threshold: float,
) -> np.ndarray:
    use_first = observed_motion >= threshold
    return np.where(use_first[:, None], first_error, blended)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--first-cache", required=True)
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--minimum-first-weight", type=float, default=0.5)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--maximum-high-motion-regression", type=float, default=0.002)
    parser.add_argument("--minimum-holdout-improvement", type=float, default=0.005)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 4 or not 0 < args.weight_step <= 1:
        raise SystemExit("samples must be at least four and weight step must be in (0,1]")
    if not 0 <= args.minimum_first_weight <= 1:
        raise SystemExit("minimum first weight must be in [0,1]")
    if args.maximum_high_motion_regression < 0 or args.minimum_holdout_improvement < 0:
        raise SystemExit("promotion tolerances must be non-negative")

    split = json.loads(Path(args.split_manifest).read_text())
    paths = [
        path
        for episode in split["validation_episodes"]
        for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))
    ]
    selected_paths = evenly_spaced(paths, args.samples)
    names = [path.name for path in selected_paths]
    contexts, targets = [], []
    for path in selected_paths:
        with np.load(path, allow_pickle=False) as data:
            contexts.append(data["context_frames"])
            targets.append(data["target_frames"])
    context, target = np.stack(contexts), np.stack(targets)
    first = load_cache(Path(args.first_cache), names, target.shape)
    second = load_cache(Path(args.second_cache), names, target.shape)
    first_error = prediction_error(first, target)
    second_error = prediction_error(second, target)
    high_motion = motion_mask(context, target, args.high_motion_threshold)
    tuning = np.arange(len(target)) % 2 == 0
    holdout = ~tuning
    all_samples = np.ones(len(target), dtype=bool)

    weights = np.arange(args.minimum_first_weight, 1.0 + args.weight_step / 2.0, args.weight_step)
    weights = np.unique(np.append(weights.clip(0, 1), 1.0))
    errors_by_weight = [blended_error(first, second, target, np.full(8, weight)) for weight in weights]
    baseline_tuning = metrics(first_error, tuning, high_motion)
    maximum_high_motion_mae = baseline_tuning["high_motion_mae_mean"] * (
        1.0 + args.maximum_high_motion_regression
    )

    candidates = []
    for statistic in ("mean", "last"):
        observed = context_motion(context, statistic)
        quantiles = np.quantile(observed[tuning], np.linspace(0.0, 1.0, 21))
        thresholds = np.unique(
            np.concatenate(
                (
                    [np.nextafter(float(observed.min()), -np.inf)],
                    quantiles,
                    [np.nextafter(float(observed.max()), np.inf)],
                )
            )
        )
        for weight_index, weight in enumerate(weights):
            for threshold in thresholds:
                error = gated_error(first_error, errors_by_weight[weight_index], observed, float(threshold))
                score = metrics(error, tuning, high_motion)
                candidates.append(
                    {
                        "context_motion_statistic": statistic,
                        "context_motion_threshold": float(threshold),
                        "low_motion_first_weight": float(weight),
                        "tuning_mae_mean": score["mae_mean"],
                        "tuning_high_motion_mae_mean": score["high_motion_mae_mean"],
                        "tuning_high_motion_constraint_passed": (
                            score["high_motion_mae_mean"] <= maximum_high_motion_mae
                        ),
                    }
                )

    eligible = [candidate for candidate in candidates if candidate["tuning_high_motion_constraint_passed"]]
    selected = min(eligible or candidates, key=lambda candidate: candidate["tuning_mae_mean"])
    selected_weight_index = int(
        np.argmin(np.abs(weights - float(selected["low_motion_first_weight"])))
    )
    selected_observed = context_motion(context, selected["context_motion_statistic"])
    selected_error = gated_error(
        first_error,
        errors_by_weight[selected_weight_index],
        selected_observed,
        float(selected["context_motion_threshold"]),
    )
    use_first = selected_observed >= float(selected["context_motion_threshold"])
    first_holdout = metrics(first_error, holdout, high_motion)
    selected_holdout = metrics(selected_error, holdout, high_motion)
    holdout_relative_improvement = (
        first_holdout["mae_mean"] - selected_holdout["mae_mean"]
    ) / first_holdout["mae_mean"]
    holdout_high_motion_relative_improvement = (
        first_holdout["high_motion_mae_mean"] - selected_holdout["high_motion_mae_mean"]
    ) / first_holdout["high_motion_mae_mean"]
    promote = (
        holdout_relative_improvement >= args.minimum_holdout_improvement
        and holdout_high_motion_relative_improvement >= -args.maximum_high_motion_regression
    )

    result = {
        "format": "track2-crossvalidated-motion-gated-blend-v1",
        "first_name": args.first_name,
        "second_name": args.second_name,
        "sample_count": len(target),
        "tuning_indices": np.flatnonzero(tuning).tolist(),
        "holdout_indices": np.flatnonzero(holdout).tolist(),
        "high_motion_sample_count": int(high_motion.sum()),
        "selection_constraint": {
            "maximum_high_motion_regression": args.maximum_high_motion_regression,
            "minimum_holdout_improvement": args.minimum_holdout_improvement,
        },
        "selected": selected,
        "gate_usage": {
            "all_use_first_only_count": int(use_first.sum()),
            "all_use_low_motion_blend_count": int((~use_first).sum()),
            "holdout_use_first_only_count": int((use_first & holdout).sum()),
            "holdout_use_low_motion_blend_count": int((~use_first & holdout).sum()),
        },
        "context_motion_diagnostic": {
            "selected_statistic": selected["context_motion_statistic"],
            "future_high_motion_observed_mean": float(selected_observed[high_motion].mean()),
            "future_low_motion_observed_mean": float(selected_observed[~high_motion].mean()),
            "observed_future_motion_correlation": float(
                np.corrcoef(selected_observed, motion_mask_score(context, target))[0, 1]
            ),
        },
        "first": {
            "all": metrics(first_error, all_samples, high_motion),
            "holdout": first_holdout,
        },
        "second": {
            "all": metrics(second_error, all_samples, high_motion),
            "holdout": metrics(second_error, holdout, high_motion),
        },
        "motion_gated_blend": {
            "all": metrics(selected_error, all_samples, high_motion),
            "holdout": selected_holdout,
        },
        "holdout_relative_improvement": holdout_relative_improvement,
        "holdout_high_motion_relative_improvement": holdout_high_motion_relative_improvement,
        "promote_to_backend": promote,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def motion_mask_score(context: np.ndarray, target: np.ndarray) -> np.ndarray:
    previous = np.concatenate((context[:, -1:], target[:, :-1]), axis=1)
    difference = np.abs(target.astype(np.int16) - previous.astype(np.int16))
    return difference.mean(axis=(1, 2, 3, 4), dtype=np.float64) / 255.0


if __name__ == "__main__":
    main()
