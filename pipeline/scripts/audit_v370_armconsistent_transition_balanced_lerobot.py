#!/usr/bin/env python3
"""Independently reconstruct and audit every v370 state/action training row."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def classify(actions: np.ndarray) -> str:
    path = [
        float(np.linalg.norm(np.diff(actions[:, start : start + 6], axis=0), axis=1).sum())
        for start in (0, 7)
    ]
    return "right" if path[1] > path[0] else "left"


def mentions_arm(task: str, arm: str) -> bool:
    return re.search(rf"\b{re.escape(arm)}\s+arm\b", task, flags=re.IGNORECASE) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    summary = json.loads((args.dataset / "conversion_summary.json").read_text())
    split = json.loads(Path(summary["split"]).read_text())
    records = summary["records"]
    expected_train = [int(value) for value in split["train_episodes"]]

    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    image_count = 0
    image_column_present_all = True
    for path in sorted((args.dataset / "data").glob("chunk-*/*.parquet")):
        parquet = pq.ParquetFile(path)
        image_count += parquet.metadata.num_rows
        image_column_present_all &= "observation.images.cam_high" in parquet.schema_arrow.names
        table = parquet.read(columns=["observation.state", "action", "episode_index"])
        states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        actions.append(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        episode_indices.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
    if not actions:
        raise ValueError("dataset has no parquet rows")
    all_actions = np.concatenate(actions)
    all_states = np.concatenate(states)
    all_episode_indices = np.concatenate(episode_indices)

    exact_errors: list[float] = []
    arms_consistent: list[bool] = []
    source = Path(summary["source"])
    for dataset_episode, record in enumerate(records):
        mask = all_episode_indices == dataset_episode
        actual_action = all_actions[mask]
        actual_state = all_states[mask]
        with h5py.File(source / "data" / f"episode{record['source_episode']}.hdf5", "r") as item:
            raw = np.asarray(item["joint_action/vector"], dtype=np.float32)
        start = int(record["selected_start"])
        stop = int(record["selected_stop_exclusive"])
        expected_action = raw[start:stop]
        expected_state = raw[start - 1 : stop - 1]
        if actual_action.shape != expected_action.shape or actual_state.shape != expected_state.shape:
            raise ValueError(f"shape mismatch in dataset episode {dataset_episode}")
        exact_errors.append(float(max(
            np.max(np.abs(actual_action - expected_action)),
            np.max(np.abs(actual_state - expected_state)),
        )))
        arm = classify(raw)
        other = "left" if arm == "right" else "right"
        arms_consistent.append(
            arm == record["arm"] and mentions_arm(record["task"], arm)
            and not mentions_arm(record["task"], other)
        )

    source_episode_set = sorted({int(r["source_episode"]) for r in records})
    right_records = [r for r in records if r["arm"] == "right"]
    checks = {
        "frozen_train40_only": source_episode_set == sorted(expected_train),
        "expected_source_arm_counts": (
            len(summary["source_left_episodes"]) == 25
            and len(summary["source_right_episodes"]) == 15
        ),
        "expected_dataset_episode_counts": (
            len(records) == summary["dataset_episodes"] == 85
            and summary["left_dataset_episodes"] == 25
            and summary["right_dataset_episodes"] == 60
        ),
        "right_sources_repeated_exactly_four": all(
            sum(int(r["source_episode"]) == episode for r in right_records) == 4
            for episode in summary["source_right_episodes"]
        ),
        "all_tasks_explicit_and_arm_consistent": all(arms_consistent),
        "finite": bool(np.isfinite(all_actions).all() and np.isfinite(all_states).all()),
        "exact_state_action_reconstruction": max(exact_errors) == 0.0,
        "row_count_matches_summary": len(all_actions) == summary["total_frames"],
        "embedded_image_column_matches_frames": image_column_present_all and image_count == len(all_actions),
        "no_forbidden_access": (
            summary["forbidden_episodes_read"] == []
            and not summary["reserved_or_hidden_evaluation_access"]
            and not summary["real_competition_submission"]
        ),
    }
    report = {
        "format": "strict-track2-v370-armconsistent-transition-balanced-audit-v1",
        "dataset": str(args.dataset.resolve()),
        "source_episodes": source_episode_set,
        "dataset_episodes": len(records),
        "frames": int(len(all_actions)),
        "left_frames": summary["left_frames"],
        "right_frames": summary["right_frames"],
        "image_count": image_count,
        "max_state_action_reconstruction_error": max(exact_errors),
        "checks": checks,
        "accepted": all(checks.values()),
        "source_public_only": True,
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 5)


if __name__ == "__main__":
    main()
