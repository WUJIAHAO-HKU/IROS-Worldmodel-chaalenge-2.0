#!/usr/bin/env python3
"""Derive a deterministic left-to-right Aloha action calibration.

Only the supplied demonstration action vectors are used. Episodes are
classified by the arm whose joints move, resampled to a common task phase,
and averaged before fitting one affine transform per joint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def resample(values: np.ndarray, length: int) -> np.ndarray:
    position = np.linspace(0, len(values) - 1, length)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight[:, None]) + values[upper] * weight[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase-steps", type=int, default=128)
    args = parser.parse_args()
    left, right = [], []
    episodes = []
    paths = sorted(args.data.glob("episode*.hdf5"), key=lambda p: int(p.stem[7:]))
    for path in paths:
        with h5py.File(path) as handle:
            action = handle["joint_action/vector"][:]
        left_motion = float(np.abs(np.diff(action[:, :7], axis=0)).mean())
        right_motion = float(np.abs(np.diff(action[:, 7:], axis=0)).mean())
        arm = "right" if right_motion > left_motion else "left"
        active = action[:, 7:] if arm == "right" else action[:, :7]
        (right if arm == "right" else left).append(resample(active, args.phase_steps))
        episodes.append({"episode": int(path.stem[7:]), "arm": arm,
                         "left_motion": left_motion, "right_motion": right_motion})
    if not left or not right:
        raise RuntimeError("both left- and right-arm demonstrations are required")
    left_mean, right_mean = np.mean(left, axis=0), np.mean(right, axis=0)
    mirror_sign = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float64)
    affine_scale, affine_bias = [], []
    affine_prediction = np.empty_like(left_mean)
    for joint in range(7):
        scale, bias = np.polyfit(left_mean[:, joint], right_mean[:, joint], 1)
        affine_scale.append(float(scale)); affine_bias.append(float(bias))
        affine_prediction[:, joint] = scale * left_mean[:, joint] + bias

    def error(prediction: np.ndarray) -> dict:
        delta = prediction - right_mean
        return {"mae": float(np.abs(delta).mean()),
                "rmse": float(np.sqrt(np.square(delta).mean())),
                "mae_per_joint": np.abs(delta).mean(0).tolist()}

    report = {
        "format": "strict-track2-official-demo-arm-calibration-v1",
        "data": str(args.data.resolve()), "demonstration_count": len(paths),
        "left_episode_count": len(left), "right_episode_count": len(right),
        "real_eval_success_labels_used": False, "world_model_predictions_used": False,
        "phase_steps": args.phase_steps, "mirror_sign": mirror_sign.tolist(),
        "affine_scale": affine_scale, "affine_bias": affine_bias,
        "trajectory_alignment_error": {
            "plain_swap": error(left_mean),
            "mirror": error(left_mean * mirror_sign),
            "affine": error(affine_prediction),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
