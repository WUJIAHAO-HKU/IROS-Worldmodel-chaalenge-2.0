#!/usr/bin/env python3
"""Compare matching Track 2 evaluation reports before paying for a full sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_report(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("format") != "track2-open-loop-eval-v1":
        raise ValueError(f"not a Track 2 open-loop evaluation: {path}")
    if int(payload.get("sample_count", 0)) < 1:
        raise ValueError(f"evaluation has no samples: {path}")
    return payload


def metric(report: dict, key: str) -> float:
    value = report.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"evaluation is missing numeric {key}")
    return float(value)


def paired_window_mae(report: dict) -> tuple[list[str], np.ndarray]:
    """Return ordered per-window means so pilot gains cannot hide sample drift."""
    windows = report.get("window_acceptance")
    if not isinstance(windows, list) or len(windows) != int(report["sample_count"]):
        raise ValueError("evaluation lacks one per-window metric for every selected sample")
    names: list[str] = []
    values: list[float] = []
    for item in windows:
        if not isinstance(item, dict) or not isinstance(item.get("window"), str):
            raise ValueError("evaluation has an invalid per-window record")
        value = item.get("mean_mae_over_8_frames")
        if not isinstance(value, (int, float)):
            raise ValueError("evaluation per-window record is missing mean_mae_over_8_frames")
        names.append(item["window"])
        values.append(float(value))
    if len(set(names)) != len(names):
        raise ValueError("evaluation repeats a window")
    return names, np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a Track 2 candidate using equal held-out samples.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.01,
        help="Required overall MAE reduction before a full 682-window evaluation.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=20_000,
        help="Deterministic paired-window bootstrap samples included in the comparison report.",
    )
    args = parser.parse_args()
    if not 0 <= args.minimum_relative_improvement < 1 or args.bootstrap_resamples < 1:
        raise SystemExit("--minimum-relative-improvement must be in [0, 1)")
    baseline = load_report(Path(args.baseline))
    candidate = load_report(Path(args.candidate))
    if baseline["sample_count"] != candidate["sample_count"] or baseline.get("split") != candidate.get("split"):
        raise SystemExit("baseline and candidate must cover the same split and sample count")
    baseline_names, baseline_windows = paired_window_mae(baseline)
    candidate_names, candidate_windows = paired_window_mae(candidate)
    if baseline_names != candidate_names:
        raise SystemExit("baseline and candidate do not cover the same ordered validation windows")
    baseline_mean = metric(baseline, "model_mae_mean")
    candidate_mean = metric(candidate, "model_mae_mean")
    baseline_motion = metric(baseline, "high_motion_model_mae_mean")
    candidate_motion = metric(candidate, "high_motion_model_mae_mean")
    relative_improvement = (baseline_mean - candidate_mean) / baseline_mean
    promote = relative_improvement >= args.minimum_relative_improvement and candidate_motion <= baseline_motion
    paired_delta = baseline_windows - candidate_windows
    generator = np.random.default_rng(0)
    bootstrap_indices = generator.integers(0, len(paired_delta), size=(args.bootstrap_resamples, len(paired_delta)))
    bootstrap_delta = paired_delta[bootstrap_indices].mean(axis=1)
    result = {
        "format": "track2-pilot-comparison-v1",
        "baseline": str(Path(args.baseline).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "sample_count": baseline["sample_count"],
        "baseline_model_mae_mean": baseline_mean,
        "candidate_model_mae_mean": candidate_mean,
        "relative_model_mae_improvement": relative_improvement,
        "baseline_high_motion_mae_mean": baseline_motion,
        "candidate_high_motion_mae_mean": candidate_motion,
        "minimum_relative_improvement": args.minimum_relative_improvement,
        "paired_window_mean_mae_delta": float(paired_delta.mean()),
        "paired_window_improved_count": int((paired_delta > 0).sum()),
        "paired_window_regressed_count": int((paired_delta < 0).sum()),
        "paired_bootstrap_resamples": args.bootstrap_resamples,
        "paired_bootstrap_mean_mae_delta_ci95": [
            float(value) for value in np.quantile(bootstrap_delta, [0.025, 0.975])
        ],
        "paired_bootstrap_probability_improved": float((bootstrap_delta > 0).mean()),
        "promote_to_full_682_evaluation": promote,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
