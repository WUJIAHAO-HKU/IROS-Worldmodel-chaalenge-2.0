#!/usr/bin/env python3
"""Build public train40 plus mirrored-left augmentation as a LeRobot dataset."""

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


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)
MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


def decode(value: np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(value.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def classify(actions: np.ndarray) -> int:
    paths = [
        float(np.linalg.norm(np.diff(actions[:, start : start + 6], axis=0), axis=1).sum())
        for start in (0, 7)
    ]
    return int(paths[1] > paths[0])


def mirror_vector(vector: np.ndarray) -> np.ndarray:
    output = np.empty_like(vector, dtype=np.float32)
    output[0:7] = vector[7:14] * MIRROR_SIGN
    output[7:14] = vector[0:7] * MIRROR_SIGN
    return output


def mirror_prompt(prompt: str) -> str:
    value = re.sub(r"\bleft\b", "__track2_right__", prompt, flags=re.IGNORECASE)
    value = re.sub(r"\bright\b", "left", value, flags=re.IGNORECASE)
    return value.replace("__track2_right__", "right")


def load_instruction(reset_dir: Path, episode: int) -> str:
    reset = np.load(reset_dir / f"episode{episode}.npy", allow_pickle=True)
    for item in reset.flat:
        if isinstance(item, dict) and item.get("instruction"):
            return str(item["instruction"])
    raise ValueError(f"episode {episode} has no instruction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reset-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    audit = json.loads(args.audit.read_text())
    assert audit["accepted"] and audit["forbidden_episodes_read"] == []
    assert np.array_equal(np.asarray(audit["mirror_sign"]), MIRROR_SIGN)
    split = json.loads(args.split.read_text())
    train = [int(x) for x in split["train_episodes"]]
    assert train == audit["train_episodes"]

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "observation.images.cam_high": {
            "dtype": "image", "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/adjust_bottle_train40_mirror_balanced_v1",
        root=args.output,
        fps=10,
        robot_type="aloha",
        features=features,
        use_videos=False,
        image_writer_threads=8,
    )

    records = []
    for episode in train:
        path = args.source / "data" / f"episode{episode}.hdf5"
        task = load_instruction(args.reset_dir, episode)
        with h5py.File(path, "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
            images = item["observation/head_camera/rgb"]
            arm = classify(actions)
            for t in range(1, len(actions)):
                dataset.add_frame({
                    "observation.state": actions[t - 1],
                    "action": actions[t],
                    "observation.images.cam_high": decode(images[t]),
                    "task": task,
                })
            dataset.save_episode()
        records.append({
            "kind": "real", "source_episode": episode,
            "arm": "right" if arm else "left", "frames": len(actions) - 1,
            "task": task,
        })

    left_episodes = [r["source_episode"] for r in records if r["arm"] == "left"]
    for episode in left_episodes:
        path = args.source / "data" / f"episode{episode}.hdf5"
        source_task = load_instruction(args.reset_dir, episode)
        task = mirror_prompt(source_task)
        with h5py.File(path, "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
            images = item["observation/head_camera/rgb"]
            assert classify(actions) == 0
            for t in range(1, len(actions)):
                dataset.add_frame({
                    "observation.state": mirror_vector(actions[t - 1]),
                    "action": mirror_vector(actions[t]),
                    "observation.images.cam_high": np.ascontiguousarray(decode(images[t])[:, ::-1]),
                    "task": task,
                })
            dataset.save_episode()
        records.append({
            "kind": "mirrored_left_to_right", "source_episode": episode,
            "arm": "right", "frames": len(actions) - 1,
            "source_task": source_task, "task": task,
        })

    result = {
        "format": "strict-track2-public-train40-mirror-balanced-lerobot-v1",
        "source": str(args.source.resolve()),
        "split": str(args.split.resolve()),
        "audit": str(args.audit.resolve()),
        "output": str(args.output.resolve()),
        "repo_id": "local/adjust_bottle_train40_mirror_balanced_v1",
        "state_alignment": "state[t]=joint_action[t-1], action[t]=joint_action[t]",
        "action_order": "left7,right7",
        "mirror_sign": MIRROR_SIGN.tolist(),
        "real_episodes": 40,
        "mirrored_episodes": len(left_episodes),
        "total_episodes": len(records),
        "real_left_episodes": 25,
        "real_right_episodes": 15,
        "effective_right_episodes": 15 + len(left_episodes),
        "total_frames": int(sum(r["frames"] for r in records)),
        "records": records,
        "forbidden_episodes_read": [],
        "reserved_or_hidden_evaluation_access": False,
        "real_competition_submission": False,
    }
    (args.output / "conversion_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "total_episodes", "total_frames", "real_left_episodes",
        "real_right_episodes", "effective_right_episodes", "output",
    )}, indent=2))


if __name__ == "__main__":
    main()
