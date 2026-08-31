#!/usr/bin/env python3
"""Summarize unchanged-policy action coverage from bridge audit NPZ files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def channel_stats(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "finite": bool(np.isfinite(flat).all()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
                np.quantile(flat, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]),
            )
        },
        "count_below_0_90": int((flat < 0.90).sum()),
        "count_below_0_75": int((flat < 0.75).sum()),
        "count_below_0_50": int((flat < 0.50).sum()),
        "count_below_0_25": int((flat < 0.25).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prereg = json.loads(args.preregistration.read_text())
    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    expected = int(prereg["capture"]["expected_items"])
    expected_shape = tuple(prereg["capture"]["expected_future_action_shape"])
    if len(files) != expected:
        raise ValueError(f"expected {expected} bridge audit files, found {len(files)}")

    chunks: list[np.ndarray] = []
    actions_by_instruction: dict[str, list[np.ndarray]] = defaultdict(list)
    instructions: set[str] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as item:
            actions = np.asarray(item["future_actions"], dtype=np.float32)
            if actions.shape != expected_shape:
                raise ValueError(f"{path} has action shape {actions.shape}, expected {expected_shape}")
            if not np.isfinite(actions).all():
                raise ValueError(f"{path} contains non-finite actions")
            chunks.append(actions)
            item_instructions = json.loads(str(item["instructions_json"]))
            if len(item_instructions) != actions.shape[0]:
                raise ValueError(
                    f"{path} has {len(item_instructions)} instructions for "
                    f"batch {actions.shape[0]}"
                )
            instructions.update(item_instructions)
            for batch_index, instruction in enumerate(item_instructions):
                actions_by_instruction[str(instruction)].append(actions[batch_index])

    all_actions = np.stack(chunks, axis=0)
    left_gripper = channel_stats(all_actions[..., 6])
    right_gripper = channel_stats(all_actions[..., 13])
    report = {
        "format": "strict-track2-v211-training-action-exploration-audit-v1",
        "preregistration": str(args.preregistration),
        "audit_dir": str(args.audit_dir),
        "files": len(files),
        "action_tensor_shape": list(all_actions.shape),
        "instructions": sorted(instructions),
        "left_gripper_action_6": left_gripper,
        "right_gripper_action_13": right_gripper,
        "per_instruction": {
            instruction: {
                "action_chunks": len(instruction_chunks),
                "left_gripper_action_6": channel_stats(
                    np.stack(instruction_chunks, axis=0)[..., 6]
                ),
                "right_gripper_action_13": channel_stats(
                    np.stack(instruction_chunks, axis=0)[..., 13]
                ),
            }
            for instruction, instruction_chunks in sorted(actions_by_instruction.items())
        },
        "diagnosis": {
            "physical_right_close_sampled": right_gripper["count_below_0_50"] > 0,
            "right_close_sample_count": right_gripper["count_below_0_50"],
            "right_near_close_sample_count": right_gripper["count_below_0_75"],
            "selection_use": False,
        },
        "passed_capture_integrity": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
