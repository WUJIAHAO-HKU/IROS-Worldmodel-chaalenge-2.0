#!/usr/bin/env python3
"""Diagnose arm routing and gripper control on the reused public R0 batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def metric(text: str, name: str) -> float:
    patterns = (
        rf"'eval/{re.escape(name)}': array\(([-+0-9.eE]+)",
        rf"'eval/{re.escape(name)}': ([-+0-9.eE]+)",
    )
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            return float(match[-1])
    raise ValueError(name)


def unpack(value: float, count: int) -> list[int]:
    packed = int(round(value * count))
    return [index for index in range(count) if packed & (1 << index)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    files = sorted(args.capture.glob("policy_action_chunk_*.npy"))
    if len(files) != 25:
        raise ValueError(f"expected 25 action chunks, got {len(files)}")
    chunks = np.stack([np.load(path, allow_pickle=False) for path in files])
    actions = chunks.transpose(1, 0, 2, 3).reshape(chunks.shape[1], -1, 14)
    text = args.launcher_log.read_text(errors="replace")
    count = int(round(metric(text, "num_trajectories")))
    right_indices = unpack(metric(text, "arm_right_bitmask"), count)
    success_indices = unpack(metric(text, "success_bitmask"), count)
    grasp_indices = unpack(metric(text, "grasp_bitmask"), count)

    records = []
    for index in right_indices:
        left = actions[index, :, :7]
        right = actions[index, :, 7:]
        left_path = float(np.linalg.norm(np.diff(left[:, :6], axis=0), axis=1).sum())
        right_path = float(np.linalg.norm(np.diff(right[:, :6], axis=0), axis=1).sum())
        first_close = np.flatnonzero(right[:, 6] < 0.5)
        records.append(
            {
                "env_index": index,
                "success": index in success_indices,
                "grasp_once": index in grasp_indices,
                "active_policy_arm": "right" if right_path >= left_path else "left",
                "left_joint_path": left_path,
                "right_joint_path": right_path,
                "left_gripper_closed_fraction": float(np.mean(left[:, 6] < 0.5)),
                "right_gripper_closed_fraction": float(np.mean(right[:, 6] < 0.5)),
                "first_right_close_step": int(first_close[0]) if len(first_close) else None,
                "right_postclose_endpoint_travel": (
                    float(np.linalg.norm(right[-1, :6] - right[first_close[0], :6]))
                    if len(first_close)
                    else None
                ),
            }
        )
    misrouted = [row for row in records if row["active_policy_arm"] == "left"]
    no_right_close = [row for row in records if row["first_right_close_step"] is None]
    report = {
        "format": "strict-track2-v260-public-right-routing-failure-v1",
        "capture": str(args.capture),
        "action_chunk_shape": list(chunks.shape),
        "right_episodes": len(records),
        "right_successes": sum(row["success"] for row in records),
        "right_grasps": sum(row["grasp_once"] for row in records),
        "wrong_left_arm_routing_count": len(misrouted),
        "right_never_closed_count": len(no_right_close),
        "correct_right_routing_count": len(records) - len(misrouted),
        "records": records,
        "diagnosis": (
            "Most public right-side failures activate and close the left arm; terminal-only "
            "SFT lacks initial approach contexts and weak inactive-arm weighting does not "
            "suppress the wrong arm."
        ),
        "candidate_remediation": (
            "Train on complete public right-arm demonstrations with deployed-horizon loss, "
            "high right-gripper weight, and full inactive-left suppression."
        ),
        "public_data_only": True,
        "reserved_final128_access": False,
        "real_competition_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
