#!/usr/bin/env python3
"""Verify the train40 plus mirrored-left LeRobot dataset exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)


def mirror(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    output[:, 0:7] = values[:, 7:14] * MIRROR_SIGN
    output[:, 7:14] = values[:, 0:7] * MIRROR_SIGN
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.dataset / "conversion_summary.json").read_text())
    records = summary["records"]
    assert summary["forbidden_episodes_read"] == []
    assert summary["real_episodes"] == 40 and summary["mirrored_episodes"] == 25
    assert summary["total_episodes"] == len(records) == 65

    actions, states, episode_indices = [], [], []
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
    actions = np.concatenate(actions)
    states = np.concatenate(states)
    episode_indices = np.concatenate(episode_indices)

    exact_errors = []
    source = Path(summary["source"])
    for dataset_episode, record in enumerate(records):
        mask = episode_indices == dataset_episode
        actual_action = actions[mask]
        actual_state = states[mask]
        with h5py.File(source / "data" / f"episode{record['source_episode']}.hdf5", "r") as item:
            raw = np.asarray(item["joint_action/vector"], dtype=np.float32)
        expected_action = raw[1:]
        expected_state = raw[:-1]
        if record["kind"] == "mirrored_left_to_right":
            expected_action = mirror(expected_action)
            expected_state = mirror(expected_state)
        exact_errors.append(float(max(
            np.max(np.abs(actual_action - expected_action)),
            np.max(np.abs(actual_state - expected_state)),
        )))

    checks = {
        "expected_65_episodes": len(records) == 65,
        "expected_9367_frames": len(actions) == summary["total_frames"] == 9367,
        "finite": bool(np.isfinite(actions).all() and np.isfinite(states).all()),
        "exact_state_action_reconstruction": max(exact_errors) == 0.0,
        "embedded_image_column_matches_frames": (
            image_column_present_all and image_count == len(actions)
        ),
        "balanced_effective_arm_counts": (
            summary["real_left_episodes"] == 25
            and summary["real_right_episodes"] == 15
            and summary["effective_right_episodes"] == 40
        ),
        "source_train_split_only": summary["forbidden_episodes_read"] == [],
    }
    report = {
        "format": "strict-track2-public-mirror-balanced-lerobot-audit-v1",
        "dataset": str(args.dataset),
        "episodes": len(records),
        "frames": int(len(actions)),
        "image_count": image_count,
        "max_state_action_reconstruction_error": max(exact_errors),
        "real_left_episodes": summary["real_left_episodes"],
        "real_right_episodes": summary["real_right_episodes"],
        "effective_right_episodes": summary["effective_right_episodes"],
        "checks": checks,
        "accepted": all(checks.values()),
        "source_public_only": True,
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 5)


if __name__ == "__main__":
    main()
