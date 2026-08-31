#!/usr/bin/env python3
"""Build the frozen v439 action gate index from public train40 demos only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


MU = np.asarray([-0.330, -1.425, -1.563, 1.617, 0.494, 0.779], dtype=np.float64)
ALPHA = np.asarray([0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(args.split.read_text())
    if split.get("format") != "strict-track2-v205-public-demo-40train-10holdout-v1":
        raise RuntimeError("v439 requires the immutable public 40/10 split")
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    right_train = [episode for episode in train if arms[episode] == "right"]
    if len(train) != 40 or len(validation) != 10 or len(right_train) != 15 or set(train) & set(validation):
        raise RuntimeError("unexpected public train/holdout boundary")

    demo_rows = []
    source_sha256 = {}
    total_close_or_held = total_positive = 0
    for episode in right_train:
        prompt = prompts[episode].lower()
        if "right arm" not in prompt or "left arm" in prompt:
            raise RuntimeError(f"episode {episode} is not explicit-right")
        path = args.source_data / f"episode{episode}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        source_sha256[str(episode)] = sha256(path)
        with h5py.File(path, "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 9:
            raise RuntimeError(f"invalid joint14 public demo: {path}")
        episode_close_or_held = episode_positive = 0
        for start in range(len(actions) - 8):
            history_last = actions[start]
            future = actions[start + 1 : start + 9]
            gripper = np.concatenate((history_last[None, 13], future[:, 13]))
            closed = gripper < 0.5
            close_indices = np.flatnonzero(closed)
            held = bool(close_indices.size and np.all(closed[close_indices[0] :]))
            if not held:
                continue
            projection = float(np.dot(future[-1, 7:13] - history_last[7:13], MU))
            episode_close_or_held += 1
            episode_positive += int(projection > 0.0)
        if episode_close_or_held == 0:
            raise RuntimeError(f"right demo {episode} has no close-or-held window")
        total_close_or_held += episode_close_or_held
        total_positive += episode_positive
        demo_rows.append({
            "episode": episode,
            "frames": int(len(actions)),
            "close_or_held_windows": episode_close_or_held,
            "positive_projection_windows": episode_positive,
        })

    payload = {
        "format": "strict-track2-v439-public-right-action-projection-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "official public train40 demonstrations only",
        "split": str(args.split.resolve()),
        "split_sha256": sha256(args.split),
        "source_data": str(args.source_data.resolve()),
        "train_episodes": train,
        "validation_episodes_excluded": validation,
        "right_train_episodes": right_train,
        "right_train_demo_sha256": source_sha256,
        "right_train_summary": demo_rows,
        "close_or_held_windows": total_close_or_held,
        "positive_projection_windows": total_positive,
        "positive_projection_fraction_observational": total_positive / total_close_or_held,
        "projection": {
            "mu_right6d": MU.tolist(),
            "alpha_8": ALPHA.tolist(),
            "teacher_delta_clip": [-8, 8],
            "teacher_delta_channels": "signed RGB preserved independently",
            "right_path_strictly_greater_than_left": True,
            "history_or_future_first_close_lt": 0.5,
            "right_gripper_held_below_threshold_after_first_close": True,
            "postclose_delta_dot_mu_strictly_positive": True,
        },
        "guards": {
            "reward_read": False,
            "outcome_read": False,
            "development_or_final_data": False,
            "policy_data_read": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "right_train_episodes": len(right_train), "close_or_held_windows": total_close_or_held}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
