#!/usr/bin/env python3
"""Memory-bounded spatial routing upper bounds for two large prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--first-dir", required=True)
    parser.add_argument("--second-dir", required=True)
    parser.add_argument("--second-mode", choices=("cache", "copy-last"), default="cache")
    parser.add_argument("--blocks", default="1,2,4,8,16,32,64,256")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first_dir, second_dir = Path(args.first_dir), Path(args.second_dir)
    first = np.load(first_dir / "prediction.npy", mmap_mode="r")
    second = np.load(second_dir / "prediction.npy", mmap_mode="r") if args.second_mode == "cache" else None
    names = np.load(first_dir / "windows.npy", allow_pickle=False).astype(str)
    if second is not None:
        second_names = np.load(second_dir / "windows.npy", allow_pickle=False).astype(str)
        if first.shape != second.shape or not np.array_equal(names, second_names):
            raise ValueError("prediction arrays are not aligned")
    blocks = [int(value) for value in args.blocks.split(",")]
    if any(value < 1 or 256 % value for value in blocks):
        raise ValueError("block sizes must be positive divisors of 256")
    methods = ["first", "second", "frame_oracle"] + [f"block_{block}" for block in blocks]
    total = {method: 0.0 for method in methods}
    moving_total = {method: 0.0 for method in methods}
    selected_second = {method: 0 for method in methods[2:]}
    count = moving_count = 0
    indices = np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=np.int64)
    for completed, index in enumerate(indices, start=1):
        name = names[index]
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            previous = np.concatenate((window["context_frames"][-1:].astype(np.float32), target[:-1]), axis=0)
            copy_last = np.broadcast_to(window["context_frames"][-1:], target.shape)
        a = first[index].astype(np.float32)
        b = copy_last.astype(np.float32) if second is None else second[index].astype(np.float32)
        a_abs, b_abs = np.abs(a - target), np.abs(b - target)
        a_error, b_error = a_abs.mean(axis=-1), b_abs.mean(axis=-1)
        moving = np.abs(target - previous).mean(axis=-1) >= args.motion_threshold * 255.0
        total["first"] += float(a_abs.sum(dtype=np.float64))
        total["second"] += float(b_abs.sum(dtype=np.float64))
        moving_total["first"] += float((a_abs * moving[..., None]).sum(dtype=np.float64))
        moving_total["second"] += float((b_abs * moving[..., None]).sum(dtype=np.float64))
        frame_choose = b_error.mean(axis=(1, 2)) < a_error.mean(axis=(1, 2))
        frame_error = np.where(frame_choose[:, None, None, None], b_abs, a_abs)
        total["frame_oracle"] += float(frame_error.sum(dtype=np.float64))
        moving_total["frame_oracle"] += float((frame_error * moving[..., None]).sum(dtype=np.float64))
        selected_second["frame_oracle"] += int(frame_choose.sum())
        for block in blocks:
            key = f"block_{block}"
            ah = a_error.reshape(8, 256 // block, block, 256 // block, block).mean(axis=(2, 4))
            bh = b_error.reshape(8, 256 // block, block, 256 // block, block).mean(axis=(2, 4))
            choose = bh < ah
            mask = np.repeat(np.repeat(choose, block, axis=1), block, axis=2)
            error = np.where(mask[..., None], b_abs, a_abs)
            total[key] += float(error.sum(dtype=np.float64))
            moving_total[key] += float((error * moving[..., None]).sum(dtype=np.float64))
            selected_second[key] += int(choose.sum())
        count += int(a_abs.size)
        moving_count += int(moving.sum()) * 3
        if completed % 16 == 0:
            print(json.dumps({"completed": completed, "total": len(indices)}), flush=True)
    first_mae = total["first"] / count
    result = {
        "format": "track2-spatial-routing-block-oracle-v1",
        "sample_count": len(indices),
        "motion_threshold": args.motion_threshold,
        "metrics": {
            method: {
                "rgb_mae": total[method] / count,
                "relative_rgb_improvement_over_first_percent": 100.0 * (first_mae - total[method] / count) / first_mae,
                "moving_rgb_mae": moving_total[method] / moving_count,
                "selected_second_regions": selected_second.get(method),
            }
            for method in methods
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
