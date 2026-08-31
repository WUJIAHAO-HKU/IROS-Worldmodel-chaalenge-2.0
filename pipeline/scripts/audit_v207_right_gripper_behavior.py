#!/usr/bin/env python3
"""Record the public-only action-level diagnosis behind the v208 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "fraction_below_0_5": float(np.mean(values < 0.5)),
        "first": float(values[0]),
        "last": float(values[-1]),
    }


def active_arm(actions: np.ndarray) -> str:
    delta = np.abs(np.diff(actions.astype(np.float32), axis=0))
    left = float(delta[:, :7].sum())
    right = float(delta[:, 7:].sum())
    return "right" if right > left else "left"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--public-hdf5-root", required=True, type=Path)
    parser.add_argument("--public-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite right-gripper diagnosis")

    action_paths = sorted((args.capture_dir / "actions").glob("policy_action_chunk_*.npy"))
    if len(action_paths) != 25:
        raise ValueError(f"expected 25 captured 8-action chunks, found {len(action_paths)}")
    policy = np.concatenate(
        [np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1, 14)[:8]
         for path in action_paths],
        axis=0,
    )
    if policy.shape != (200, 14) or not np.isfinite(policy).all():
        raise ValueError(f"invalid captured policy action shape {policy.shape}")

    launcher = (args.capture_dir / "launcher.log").read_text(errors="replace")
    arm_match = re.findall(r"'eval/arm_right': array\(([^,]+)", launcher)
    success_match = re.findall(r"'eval/right_success': array\(([^,]+)", launcher)
    if not arm_match or float(arm_match[-1]) != 1.0:
        raise ValueError("capture is not an authoritative public right-arm trajectory")
    if not success_match or float(success_match[-1]) != 0.0:
        raise ValueError("capture is not the expected failed right-arm trajectory")

    split = json.loads(args.public_split.read_text())
    right_episodes = [
        int(value) for value in split["train_episodes"]
        if split["arm_by_episode"][str(value)] == "right"
    ]
    demo_actions = []
    for episode in right_episodes:
        path = args.public_hdf5_root / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as handle:
            actions = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        if active_arm(actions) != "right":
            raise ValueError(f"public episode {episode} is not right-arm active")
        demo_actions.append(actions)
    demo = np.concatenate(demo_actions, axis=0)

    report = {
        "format": "strict-track2-v207-public-right-gripper-diagnosis-v1",
        "capture": {
            "directory": str(args.capture_dir.resolve()),
            "launcher_sha256": sha256(args.capture_dir / "launcher.log"),
            "chunk_count": len(action_paths),
            "action_count": int(policy.shape[0]),
            "seed": 183,
            "arm": "right",
            "success": False,
            "left_gripper_action_6": stats(policy[:, 6]),
            "right_gripper_action_13": stats(policy[:, 13]),
        },
        "public_success_demonstrations": {
            "source": str(args.public_hdf5_root.resolve()),
            "split": str(args.public_split.resolve()),
            "split_sha256": sha256(args.public_split),
            "right_train_episodes": right_episodes,
            "right_gripper_action_13": stats(demo[:, 13]),
            "left_gripper_action_6": stats(demo[:, 6]),
        },
        "checks": {},
        "data_guard": (
            "one frozen public-development seed and declared public training demonstrations only; "
            "no hidden/final outcome and no action modification"
        ),
        "real_submission_performed": False,
    }
    checks = {
        "policy_right_gripper_stays_open":
            report["capture"]["right_gripper_action_13"]["fraction_below_0_5"] <= 0.05,
        "policy_closes_opposite_gripper":
            report["capture"]["left_gripper_action_6"]["fraction_below_0_5"] >= 0.25,
        "successful_right_demos_close_right_gripper":
            report["public_success_demonstrations"]["right_gripper_action_13"]["fraction_below_0_5"] >= 0.25,
        "successful_right_demos_leave_left_gripper_open":
            report["public_success_demonstrations"]["left_gripper_action_6"]["fraction_below_0_5"] <= 0.05,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("right-gripper diagnosis did not satisfy its fixed checks")


if __name__ == "__main__":
    main()
