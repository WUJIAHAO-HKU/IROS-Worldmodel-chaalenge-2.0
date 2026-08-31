#!/usr/bin/env python3
"""Audit randomized RoboTwin HDF5 files before v26 pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from wam_pipeline.external_robotwin_data_v260 import ACTION_DIM, TOTAL_FRAMES, episode_identifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be positive")
    root = Path(args.root)
    records = []
    invalid = []
    action_min = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    action_max = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    arm_counts = {"left": 0, "right": 0, "both": 0, "static": 0}
    total_frames = 0
    total_windows = 0
    for path in sorted(root.glob("episode_*/episode_*.hdf5"), key=episode_identifier):
        try:
            with h5py.File(path, "r") as handle:
                rgb = handle["observations/images/cam_high"]
                actions = np.asarray(handle["action"], dtype=np.float32)
                if actions.shape != (len(rgb), ACTION_DIM) or len(rgb) < TOTAL_FRAMES:
                    raise ValueError(f"rgb/actions shape mismatch: {len(rgb)} vs {actions.shape}")
                if not np.isfinite(actions).all():
                    raise ValueError("non-finite action")
                # Decoding is covered by the dataset smoke; touching both ends
                # here catches truncated variable-length HDF5 payloads cheaply.
                if len(rgb[0]) == 0 or len(rgb[-1]) == 0:
                    raise ValueError("empty head-camera payload")
                frame_count = len(rgb)
        except Exception as exc:
            invalid.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        identifier = episode_identifier(path)
        left_span = float(np.ptp(actions[:, :6], axis=0).max())
        right_span = float(np.ptp(actions[:, 7:13], axis=0).max())
        key = {(True, False): "left", (False, True): "right", (True, True): "both", (False, False): "static"}[
            (left_span > 1e-4, right_span > 1e-4)
        ]
        arm_counts[key] += 1
        windows = 1 + (frame_count - TOTAL_FRAMES) // args.stride
        if (frame_count - TOTAL_FRAMES) % args.stride:
            windows += 1
        total_frames += frame_count
        total_windows += windows
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        records.append(
            {
                "episode": identifier,
                "frames": frame_count,
                "stride_windows": windows,
                "active_arm": key,
                "left_joint_span": left_span,
                "right_joint_span": right_span,
            }
        )
    report = {
        "format": "track2-external-randomized-data-audit-v26.0",
        "source": {
            "repository": "sunLry/robotwin-random-47-500",
            "path": "adjust_bottle-demo_randomized-1000",
            "url": "https://huggingface.co/datasets/sunLry/robotwin-random-47-500",
            "declared_license": "other",
        },
        "root": str(root.resolve()),
        "stride": args.stride,
        "valid_episode_count": len(records),
        "invalid_episode_count": len(invalid),
        "total_frames": total_frames,
        "total_stride_windows": total_windows,
        "active_arm_counts": arm_counts,
        "action_min": action_min.tolist() if records else None,
        "action_max": action_max.tolist() if records else None,
        "episodes": records,
        "invalid": invalid,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in ("valid_episode_count", "invalid_episode_count", "total_frames", "total_stride_windows", "active_arm_counts")}, indent=2))
    if invalid:
        raise SystemExit("external dataset audit found invalid files")


if __name__ == "__main__":
    main()
