#!/usr/bin/env python3
"""Audit mixed-arm labels inferred from public mirror-balanced action chunks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--gripper-activity-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    summary = json.loads((args.dataset / "conversion_summary.json").read_text())
    records = summary["records"]
    stats = json.loads(args.norm_stats.read_text())["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    if q01.shape != q99.shape or q01.shape[0] != 14:
        raise RuntimeError("expected official 14D action quantiles")
    normalized_zero = np.clip(2.0 * (np.zeros(14) - q01) / (q99 - q01) - 1.0, -1.0, 1.0)

    rows = []
    for dataset_episode, record in enumerate(records):
        path = args.dataset / "data" / "chunk-000" / f"episode_{dataset_episode:06d}.parquet"
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        if actions.shape != states.shape or actions.shape[1] != 14:
            raise RuntimeError(f"invalid action/state shape in {path}")
        correct = 0
        right_predictions = 0
        margins = []
        for frame in range(len(actions)):
            indices = np.minimum(
                frame + np.arange(args.horizon, dtype=np.int64), len(actions) - 1
            )
            chunk = actions[indices].copy()
            chunk[:, :6] -= states[frame, :6]
            chunk[:, 7:13] -= states[frame, 7:13]
            normalized = np.clip(2.0 * (chunk - q01) / (q99 - q01) - 1.0, -1.0, 1.0)
            left_joint = np.abs(normalized[:, :6] - normalized_zero[:6]).mean()
            right_joint = np.abs(normalized[:, 7:13] - normalized_zero[7:13]).mean()
            left_close = (0.5 * (1.0 - np.clip(normalized[:, 6], -1.0, 1.0))).mean()
            right_close = (0.5 * (1.0 - np.clip(normalized[:, 13], -1.0, 1.0))).mean()
            left_score = left_joint + args.gripper_activity_weight * left_close
            right_score = right_joint + args.gripper_activity_weight * right_close
            predicted = "right" if right_score > left_score else "left"
            correct += int(predicted == record["arm"])
            right_predictions += int(predicted == "right")
            margins.append(abs(right_score - left_score))
        rows.append(
            {
                "dataset_episode": dataset_episode,
                "kind": record["kind"],
                "source_episode": record["source_episode"],
                "expected_arm": record["arm"],
                "frames": len(actions),
                "correct": correct,
                "accuracy": correct / len(actions),
                "right_predictions": right_predictions,
                "mean_score_margin": float(np.mean(margins)),
                "min_score_margin": float(np.min(margins)),
            }
        )

    total = sum(row["frames"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    left_rows = [row for row in rows if row["expected_arm"] == "left"]
    right_rows = [row for row in rows if row["expected_arm"] == "right"]
    left_correct = sum(row["correct"] for row in left_rows)
    left_total = sum(row["frames"] for row in left_rows)
    right_correct = sum(row["correct"] for row in right_rows)
    right_total = sum(row["frames"] for row in right_rows)
    checks = {
        "all_65_public_augmented_episodes_read": len(rows) == 65,
        "all_9367_frames_read": total == 9367,
        "overall_accuracy_at_least_98pct": correct / total >= 0.98,
        "left_accuracy_at_least_98pct": left_correct / left_total >= 0.98,
        "right_accuracy_at_least_98pct": right_correct / right_total >= 0.98,
    }
    report = {
        "format": "strict-track2-mixed-arm-action-inference-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "norm_stats": str(args.norm_stats),
        "horizon": args.horizon,
        "gripper_activity_weight": args.gripper_activity_weight,
        "total_frames": total,
        "overall_accuracy": correct / total,
        "left_accuracy": left_correct / left_total,
        "right_accuracy": right_correct / right_total,
        "checks": checks,
        "accepted": all(checks.values()),
        "records": rows,
        "public_data_only": True,
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "total_frames", "overall_accuracy", "left_accuracy", "right_accuracy", "checks", "accepted"
    )}, indent=2))
    raise SystemExit(0 if report["accepted"] else 5)


if __name__ == "__main__":
    main()
