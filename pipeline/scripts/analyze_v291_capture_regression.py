#!/usr/bin/env python3
"""Explain the public capture reward differences between v271 and v290."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wam_pipeline.arm_routed_autoregressive_runtime import (
    Track2ArmRoutedAutoregressiveUNet,
)


def path_length(history: np.ndarray, future: np.ndarray) -> float:
    sequence = np.concatenate([history[-1:], future], axis=0)
    return float(np.linalg.norm(np.diff(sequence[:, 7:13], axis=0), axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--v271-details", type=Path, required=True)
    parser.add_argument("--v290-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records: list[dict] = []
    for file_index, path in enumerate(sorted(args.audit_dir.glob("rollout_*.npz"))):
        with np.load(path, allow_pickle=False) as data:
            history = data["history_actions"].astype(np.float32)
            future = data["future_actions"].astype(np.float32)
            texts = list(map(str, json.loads(str(data["instructions_json"]))))
        for action_index, (h, f, text) in enumerate(zip(history, future, texts, strict=True)):
            sequence = np.concatenate([h[-1:], f], axis=0)
            delta = np.abs(np.diff(sequence, axis=0))
            left_motion = float(delta[:, :6].mean())
            right_motion = float(delta[:, 7:13].mean())
            # Reproduce the exact route and post-grasp filter from the audit.
            route = Track2ArmRoutedAutoregressiveUNet.active_arm(h, f, text)
            if route != "right":
                continue
            if h[-1, 13] >= 0.5 or float((f[:, 13] < 0.5).mean()) < 0.75:
                continue
            records.append(
                {
                    "file_index": file_index,
                    "action_index": action_index,
                    "instruction": text,
                    "history_right_gripper": float(h[-1, 13]),
                    "future_right_gripper_mean": float(f[:, 13].mean()),
                    "future_right_gripper_min": float(f[:, 13].min()),
                    "future_right_gripper_max": float(f[:, 13].max()),
                    "left_motion_mean": left_motion,
                    "right_motion_mean": right_motion,
                    "right_path_length": path_length(h, f),
                    "right_endpoint_norm": float(np.linalg.norm(f[-1, 7:13] - h[-1, 7:13])),
                }
            )

    with np.load(args.v271_details, allow_pickle=False) as data:
        v271 = data["rewards"][:, -1].astype(float)
        alpha = data["alpha"].astype(float)
        alignment = data["alignment"].astype(float)
        delta_ratio = data["delta_ratio"].astype(float)
    with np.load(args.v290_details, allow_pickle=False) as data:
        v290 = data["rewards"][:, -1].astype(float)
    if len(records) != len(v271) or v271.shape != v290.shape:
        raise RuntimeError((len(records), v271.shape, v290.shape))

    for index, row in enumerate(records):
        row.update(
            {
                "record_index": index,
                "v271_terminal_reward": float(v271[index]),
                "v290_terminal_reward": float(v290[index]),
                "reward_change": float(v290[index] - v271[index]),
                "alpha": float(alpha[index]),
                "alignment": float(alignment[index]),
                "delta_ratio": float(delta_ratio[index]),
            }
        )
    changed = sorted(records, key=lambda row: abs(row["reward_change"]), reverse=True)
    report = {
        "format": "strict-track2-v291-capture-regression-analysis-v1",
        "records": len(records),
        "v271_success_like_indices": np.flatnonzero(v271 >= 0.1).tolist(),
        "v290_success_like_indices": np.flatnonzero(v290 >= 0.1).tolist(),
        "success_like_lost": np.flatnonzero((v271 >= 0.1) & (v290 < 0.1)).tolist(),
        "success_like_gained": np.flatnonzero((v271 < 0.1) & (v290 >= 0.1)).tolist(),
        "mean_reward_v271": float(v271.mean()),
        "mean_reward_v290": float(v290.mean()),
        "mean_absolute_reward_change": float(np.abs(v290 - v271).mean()),
        "largest_changes": changed[:24],
        "v271_success_like_rows": [records[i] for i in np.flatnonzero(v271 >= 0.1)],
        "v290_success_like_rows": [records[i] for i in np.flatnonzero(v290 >= 0.1)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
