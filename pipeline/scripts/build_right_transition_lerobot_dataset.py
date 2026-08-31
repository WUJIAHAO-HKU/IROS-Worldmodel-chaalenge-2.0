#!/usr/bin/env python3
"""Build public right-arm transition windows without static terminal oversampling."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]


def decode_rgb(value: np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(value.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


summary_path, output_path = map(Path, sys.argv[1:3])
if output_path.exists():
    raise FileExistsError(f"refusing to overwrite {output_path}")
summary = json.loads(summary_path.read_text())
source = Path(summary["source"])
features = {
    "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
    "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
    "observation.images.cam_high": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
}
dataset = LeRobotDataset.create(
    repo_id="local/adjust_bottle_train40_right_transition",
    root=output_path,
    fps=10,
    robot_type="aloha",
    features=features,
    use_videos=False,
    image_writer_threads=8,
)
records = []
for ordinal, episode_item in enumerate(summary["episode_summaries"], start=1):
    episode = int(episode_item["source_episode"])
    task = str(episode_item["task"])
    with h5py.File(source / "data" / f"episode{episode}.hdf5", "r") as item:
        actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
        images = item["observation/head_camera/rgb"]
        closed = np.flatnonzero(actions[:, 13] < 0.5)
        if not closed.size:
            raise ValueError(f"right gripper never closes in episode {episode}")
        first_close = int(closed[0])
        start = max(1, first_close - 4)
        stop = min(len(actions), first_close + 49)
        for t in range(start, stop):
            dataset.add_frame({
                "observation.state": actions[t - 1],
                "action": actions[t],
                "observation.images.cam_high": decode_rgb(images[t]),
                "task": task,
            })
        dataset.save_episode()
    records.append({
        "source_episode": episode,
        "task": task,
        "source_frames": len(actions),
        "first_right_close": first_close,
        "selected_start": start,
        "selected_stop_exclusive": stop,
        "selected_frames": stop - start,
        "postclose_frames": stop - first_close,
    })
    print(f"[{ordinal}/{len(summary['episode_summaries'])}] episode={episode}", flush=True)
result = {
    "format": "track2-public-right-grasp-motion-transition-sft-v1",
    "source_conversion_summary": str(summary_path),
    "source": str(source),
    "output": str(output_path),
    "repo_id": "local/adjust_bottle_train40_right_transition",
    "selection": "frames from 4 steps before first right close through 48 steps after close",
    "rationale": "retain close/lift/place transitions while removing most static terminal hold frames",
    "state_alignment": "state[t]=joint_action[t-1], action[t]=joint_action[t]",
    "action_order": "left7,right7",
    "episodes": len(records),
    "frames": sum(item["selected_frames"] for item in records),
    "records": records,
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
(output_path / "conversion_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
