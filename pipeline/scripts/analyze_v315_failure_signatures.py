#!/usr/bin/env python3
"""Diagnose sparse failure signatures from the frozen v314 audit.

This is an exploratory, read-only analysis.  The validation split may be used
to choose a future preregistered rule; local_test is emitted only as an
episode-disjoint confirmation and must not influence that choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


THRESHOLDS = (1e-4, 1e-3, 1e-2, 1e-1)


def action_stats(history: np.ndarray, future: np.ndarray) -> dict[str, float | bool]:
    sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
    step = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    return {
        "path": float(step.sum()),
        "net": float(np.linalg.norm(future[-1, 7:13] - history[-1, 7:13])),
        "history_right_gripper": float(history[-1, 13]),
        "future_right_gripper_mean": float(future[:, 13].mean()),
        "release_after_closed": bool(history[-1, 13] <= 0.5 and future[:, 13].mean() > 0.5),
        "static": bool(step.sum() <= 1e-6),
    }


def summarize(rows: list[dict], threshold: float) -> dict:
    valid_failures = []
    for row in rows:
        stats = row["action_stats"]
        reverse_like = (
            row["positive_gate_probability"] < threshold
            and stats["future_right_gripper_mean"] <= 0.5
            and stats["path"] > 1e-6
        )
        if stats["release_after_closed"] or stats["static"] or reverse_like:
            valid_failures.append(row)

    transition = [
        row
        for row in rows
        if row["context_terminal_reward"] <= 0.10 and row["gt_terminal_reward"] >= 0.90
    ]
    kinds = {}
    for kind in ("open_gripper", "static_transport", "reverse_transport"):
        detected = []
        for row in transition:
            stats = row["action_stats"]
            if kind == "open_gripper":
                value = stats["history_right_gripper"] <= 0.5
            elif kind == "static_transport":
                value = True
            else:
                probability = row["counterfactual_gate"][kind]["probability"]
                value = probability is not None and probability < threshold
            detected.append(bool(value))
        kinds[kind] = float(np.mean(detected)) if detected else 0.0

    false_count = len(valid_failures)
    return {
        "threshold": threshold,
        "valid_windows": len(rows),
        "valid_failure_signature_count": false_count,
        "valid_failure_signature_rate": false_count / max(len(rows), 1),
        "valid_failure_windows": [row["window"] for row in valid_failures],
        "valid_signature_gt_reward_mean": (
            float(np.mean([row["gt_terminal_reward"] for row in valid_failures]))
            if valid_failures
            else None
        ),
        "valid_signature_context_to_gt_rgb_proxy": {
            "direct_terminal_mae_mean": (
                float(np.mean([row["direct_terminal_rgb_mae"] for row in valid_failures]))
                if valid_failures
                else None
            ),
            "v314_context_terminal_mae_mean": (
                float(np.mean([row["candidate_terminal_rgb_mae"] for row in valid_failures]))
                if valid_failures
                else None
            ),
        },
        "transition_windows": len(transition),
        "transition_counterfactual_detection": kinds,
        "transition_all_three_detection": float(
            np.mean([all(kinds[k] == 1.0 for k in kinds)])
        ) if transition else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v314-report", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.v314_report.read_text())
    enriched = {}
    for split_name, source_rows in report["rows"].items():
        rows = []
        for source in source_rows:
            row = dict(source)
            with np.load(args.windows / source["window"], allow_pickle=False) as data:
                row["action_stats"] = action_stats(
                    np.asarray(data["history_actions"], dtype=np.float32),
                    np.asarray(data["future_actions"], dtype=np.float32),
                )
            rows.append(row)
        enriched[split_name] = rows
    result = {
        "format": "strict-track2-v315-failure-signature-exploration-v1",
        "status": "exploratory_only_not_an_authorization_gate",
        "selection_policy": "validation may choose; local_test confirmation only",
        "threshold_summaries": {
            split_name: [summarize(rows, value) for value in THRESHOLDS]
            for split_name, rows in enriched.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
