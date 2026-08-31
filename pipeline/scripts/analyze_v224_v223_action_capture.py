#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


capture, prior_log, capture_log, conversion_summary, output = map(Path, sys.argv[1:6])


def metric(text: str, name: str) -> float:
    patterns = (
        rf"'eval/{re.escape(name)}': array\(([-+0-9.eE]+)",
        rf"'eval/{re.escape(name)}': ([-+0-9.eE]+)",
    )
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return float(found[-1])
    raise ValueError(name)


metric_names = (
    "arm_left",
    "arm_right",
    "success_once",
    "left_success",
    "right_success",
    "grasp_once",
    "left_grasp",
    "right_grasp",
    "success_bitmask",
    "grasp_bitmask",
    "arm_right_bitmask",
    "num_trajectories",
)
prior_text = prior_log.read_text(errors="replace")
capture_text = capture_log.read_text(errors="replace")
prior_metrics = {name: metric(prior_text, name) for name in metric_names}
capture_metrics = {name: metric(capture_text, name) for name in metric_names}
assert prior_metrics == capture_metrics, (prior_metrics, capture_metrics)
num_envs = int(round(capture_metrics["num_trajectories"]))


def unpack_bitmask(value: float) -> list[int]:
    packed = int(round(value * num_envs))
    return [index for index in range(num_envs) if packed & (1 << index)]


right_indices = unpack_bitmask(capture_metrics["arm_right_bitmask"])
success_indices = unpack_bitmask(capture_metrics["success_bitmask"])
grasp_indices = unpack_bitmask(capture_metrics["grasp_bitmask"])

action_files = sorted(capture.glob("policy_action_chunk_*.npy"))
assert action_files
chunks = np.stack([np.load(path) for path in action_files])
assert chunks.ndim == 4 and chunks.shape[1] == num_envs and chunks.shape[-1] == 14
actions = chunks.transpose(1, 0, 2, 3).reshape(num_envs, -1, 14)

observation_files = sorted((capture / "observations").glob("observation_*.npz"))
assert observation_files
reset_files = [path for path in observation_files if path.name.endswith("_reset.npz")]
post_files = [
    path
    for path in observation_files
    if path.name.endswith("_post_step.npz") or path.name.endswith("_post_chunk.npz")
]
assert reset_files and post_files
with np.load(reset_files[0]) as item:
    initial_states = np.asarray(item["states"], dtype=np.float32)
with np.load(post_files[-1]) as item:
    final_states = np.asarray(item["states"], dtype=np.float32)
assert initial_states.shape == final_states.shape == (num_envs, 14)

summary = json.loads(conversion_summary.read_text())
source = Path(summary["source"])
expert_endpoints = []
for episode in summary["episodes"]:
    with h5py.File(source / "data" / f"episode{episode}.hdf5", "r") as item:
        expert_actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
    expert_endpoints.append(expert_actions[-8:, 7:14].mean(axis=0))
expert_endpoints = np.stack(expert_endpoints)

records = []
for env_index in range(num_envs):
    right_action = actions[env_index, :, 7:14]
    right_joint = right_action[:, :6]
    close_steps = np.flatnonzero(right_action[:, 6] < 0.5)
    close_step = int(close_steps[0]) if close_steps.size else None
    final_command = right_action[-8:].mean(axis=0)
    endpoint_distances = np.linalg.norm(expert_endpoints - final_command, axis=1)
    postclose_travel = (
        float(np.linalg.norm(right_joint[-1] - right_joint[close_step]))
        if close_step is not None
        else None
    )
    records.append(
        {
            "env_index": env_index,
            "arm": "right" if env_index in right_indices else "left",
            "success": env_index in success_indices,
            "grasp_once": env_index in grasp_indices,
            "first_right_close_step": close_step,
            "right_close_fraction": float((right_action[:, 6] < 0.5).mean()),
            "right_joint_command_path_length": float(
                np.linalg.norm(np.diff(right_joint, axis=0), axis=1).sum()
            ),
            "right_joint_command_range_l2": float(
                np.linalg.norm(right_joint.max(axis=0) - right_joint.min(axis=0))
            ),
            "right_postclose_command_travel_l2": postclose_travel,
            "right_state_initial_to_final_l2": float(
                np.linalg.norm(final_states[env_index, 7:13] - initial_states[env_index, 7:13])
            ),
            "right_final_gripper_state": float(final_states[env_index, 13]),
            "right_final_command": final_command.tolist(),
            "right_final_command_nearest_expert_endpoint_l2": float(endpoint_distances.min()),
            "right_final_command_nearest_expert_episode": int(
                summary["episodes"][int(endpoint_distances.argmin())]
            ),
        }
    )


def aggregate(items: list[dict], key: str) -> dict:
    values = np.asarray([item[key] for item in items if item[key] is not None], dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


right_records = [item for item in records if item["arm"] == "right"]
right_grasp_records = [item for item in right_records if item["grasp_once"]]
keys = (
    "right_close_fraction",
    "right_joint_command_path_length",
    "right_joint_command_range_l2",
    "right_postclose_command_travel_l2",
    "right_state_initial_to_final_l2",
    "right_final_gripper_state",
    "right_final_command_nearest_expert_endpoint_l2",
)
report = {
    "capture_equivalent_to_prior_gate": True,
    "prior_metrics": prior_metrics,
    "action_chunk_shape": list(chunks.shape),
    "action_steps_per_env": int(actions.shape[1]),
    "observation_files": len(observation_files),
    "right_indices": right_indices,
    "success_indices": success_indices,
    "grasp_indices": grasp_indices,
    "right_all_aggregate": {key: aggregate(right_records, key) for key in keys},
    "right_grasp_aggregate": {key: aggregate(right_grasp_records, key) for key in keys},
    "records": records,
    "expert_endpoint_count": len(expert_endpoints),
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
