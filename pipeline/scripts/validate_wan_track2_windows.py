#!/usr/bin/env python3
"""Validate all exact Track 2 windows made available to a Wan trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_pipeline.profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES
from wam_pipeline.wan_track2_data import TOTAL_FRAMES, WanTrack2WindowDataset


def summarize(root: Path, logical_split: str) -> dict:
    dataset = WanTrack2WindowDataset(root, logical_split)
    per_episode: dict[str, int] = {}
    for index in range(len(dataset)):
        window = dataset[index]
        if window.context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3):
            raise ValueError(f"invalid context frame shape at index {index}")
        if window.target_frames.shape != (PREDICTION_FRAMES, 256, 256, 3):
            raise ValueError(f"invalid target frame shape at index {index}")
        if window.history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM):
            raise ValueError(f"invalid history action shape at index {index}")
        if window.future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError(f"invalid future action shape at index {index}")
        if window.video_frames.shape != (TOTAL_FRAMES, 256, 256, 3):
            raise ValueError(f"invalid full video shape at index {index}")
        slots = window.action_slots
        if slots.shape != (TOTAL_FRAMES, ACTION_DIM) or not np.array_equal(slots[0], np.zeros(ACTION_DIM, dtype=np.float32)):
            raise ValueError(f"invalid action-slot alignment at index {index}")
        if not np.array_equal(slots[1:5], window.history_actions) or not np.array_equal(slots[5:], window.future_actions):
            raise ValueError(f"history/future action slots are misaligned at index {index}")
        per_episode[window.source] = per_episode.get(window.source, 0) + 1
    return {
        "episode_count": len(per_episode),
        "window_count": len(dataset),
        "window_layout": {
            "context_frames": CONTEXT_FRAMES,
            "history_actions": CONTEXT_ACTIONS,
            "future_actions": PREDICTION_FRAMES,
            "target_frames": PREDICTION_FRAMES,
            "wan_video_frames": TOTAL_FRAMES,
            "wan_action_slots": TOTAL_FRAMES,
            "slot_zero": "zero_anchor_for_context_frame_0",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    try:
        result = {
            "format": "track2-wan-window-validation-v1",
            "status": "valid",
            "dataset_root": str(root.resolve()),
            "train": summarize(root, "train"),
            "validation": summarize(root, "validation"),
        }
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Wan Track 2 window validation failed: {exc}") from exc
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
