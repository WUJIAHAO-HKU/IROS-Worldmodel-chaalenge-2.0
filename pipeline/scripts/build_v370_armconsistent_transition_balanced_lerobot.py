#!/usr/bin/env python3
"""Build an arm-consistent, right-transition-balanced public Track-2 SFT set."""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


def decode(value: np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(value.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def classify(actions: np.ndarray) -> str:
    path = [
        float(np.linalg.norm(np.diff(actions[:, start : start + 6], axis=0), axis=1).sum())
        for start in (0, 7)
    ]
    return "right" if path[1] > path[0] else "left"


def mentions_arm(task: str, arm: str) -> bool:
    return re.search(rf"\b{re.escape(arm)}\s+arm\b", task, flags=re.IGNORECASE) is not None


def add_slice(dataset: LeRobotDataset, item: h5py.File, task: str, start: int, stop: int) -> None:
    actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
    images = item["observation/head_camera/rgb"]
    for t in range(start, stop):
        dataset.add_frame({
            "observation.state": actions[t - 1],
            "action": actions[t],
            "observation.images.cam_high": decode(images[t]),
            "task": task,
        })
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right-repeats", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.right_repeats < 1:
        raise ValueError("right-repeats must be positive")

    split = json.loads(args.split.read_text())
    train = [int(value) for value in split["train_episodes"]]
    arm_by_episode = {int(key): value for key, value in split["arm_by_episode"].items()}
    instructions = {int(key): value for key, value in split["episode_to_instruction"].items()}
    if len(train) != 40 or len(set(train)) != 40:
        raise ValueError("expected the frozen 40-episode public train split")

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "observation.images.cam_high": {
            "dtype": "image", "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/adjust_bottle_train40_armconsistent_transition_balanced_v1",
        root=args.output,
        fps=10,
        robot_type="aloha",
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )

    records: list[dict] = []
    observed_left: list[int] = []
    observed_right: list[int] = []
    for episode in train:
        path = args.source / "data" / f"episode{episode}.hdf5"
        task = str(instructions[episode])
        declared_arm = str(arm_by_episode[episode])
        if declared_arm not in ("left", "right"):
            raise ValueError(f"bad declared arm for episode {episode}: {declared_arm}")
        other_arm = "left" if declared_arm == "right" else "right"
        if not mentions_arm(task, declared_arm) or mentions_arm(task, other_arm):
            raise ValueError(f"instruction is not explicit and arm-consistent: episode={episode} task={task!r}")
        with h5py.File(path, "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
            measured_arm = classify(actions)
            if measured_arm != declared_arm:
                raise ValueError(
                    f"split/action arm mismatch episode={episode}: {declared_arm} != {measured_arm}"
                )
            if measured_arm == "left":
                start, stop = 1, len(actions)
                add_slice(dataset, item, task, start, stop)
                records.append({
                    "kind": "left_full", "source_episode": episode, "repeat": 0,
                    "arm": measured_arm, "selected_start": start,
                    "selected_stop_exclusive": stop, "frames": stop - start, "task": task,
                })
                observed_left.append(episode)
            else:
                closed = np.flatnonzero(actions[:, 13] < 0.5)
                if not closed.size:
                    raise ValueError(f"right gripper never closes in episode {episode}")
                first_close = int(closed[0])
                start = max(1, first_close - 4)
                stop = min(len(actions), first_close + 49)
                for repeat in range(args.right_repeats):
                    add_slice(dataset, item, task, start, stop)
                    records.append({
                        "kind": "right_transition", "source_episode": episode,
                        "repeat": repeat, "arm": measured_arm,
                        "first_right_close": first_close, "selected_start": start,
                        "selected_stop_exclusive": stop, "frames": stop - start,
                        "postclose_frames": stop - first_close, "task": task,
                    })
                observed_right.append(episode)

    result = {
        "format": "strict-track2-public-armconsistent-transition-balanced-lerobot-v1",
        "source": str(args.source.resolve()),
        "split": str(args.split.resolve()),
        "output": str(args.output.resolve()),
        "repo_id": "local/adjust_bottle_train40_armconsistent_transition_balanced_v1",
        "selection": {
            "left": "all frames t=1..end-1, exactly once",
            "right": "4 frames before first right close through 48 frames after close",
            "right_repeats": args.right_repeats,
        },
        "state_alignment": "state[t]=joint_action[t-1], action[t]=joint_action[t]",
        "action_order": "left7,right7",
        "source_train_episodes": train,
        "source_left_episodes": observed_left,
        "source_right_episodes": observed_right,
        "dataset_episodes": len(records),
        "left_dataset_episodes": len(observed_left),
        "right_dataset_episodes": len(observed_right) * args.right_repeats,
        "left_frames": int(sum(r["frames"] for r in records if r["arm"] == "left")),
        "right_frames": int(sum(r["frames"] for r in records if r["arm"] == "right")),
        "total_frames": int(sum(r["frames"] for r in records)),
        "records": records,
        "forbidden_episodes_read": [],
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    }
    (args.output / "conversion_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "dataset_episodes", "left_dataset_episodes", "right_dataset_episodes",
        "left_frames", "right_frames", "total_frames", "output",
    )}, indent=2))


if __name__ == "__main__":
    main()
