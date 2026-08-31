#!/usr/bin/env python3
"""Pixel equivalence and throughput audit for batched v301 versus v295."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.v295_terminal_frame_preserving_mirror_runtime import (
    Track2V295TerminalFramePreservingMirror,
)
from wam_pipeline.v301_batched_terminal_frame_mirror_runtime import (
    Track2V301BatchedTerminalFrameMirror,
)


def selected_paths(root: Path) -> list[Path]:
    # Four right-arm and four left-arm public episodes, sampled across phases.
    episodes = [7, 18, 6, 22, 5, 16, 2, 9]
    output: list[Path] = []
    for episode in episodes:
        paths = sorted(root.glob(f"episode{episode}_*.npz"))
        positions = np.linspace(0, len(paths) - 1, 8).round().astype(int)
        output.extend(paths[int(position)] for position in positions)
    return output


def load_batch(paths: list[Path]):
    values = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            values.append(
                (
                    data["context_frames"].astype(np.uint8),
                    data["history_actions"].astype(np.float32),
                    data["future_actions"].astype(np.float32),
                )
            )
    return (
        np.stack([value[0] for value in values]),
        np.stack([value[1] for value in values]),
        np.stack([value[2] for value in values]),
        np.arange(1000, 1000 + len(values), dtype=np.int64),
        [None] * len(values),
    )


def timed(runtime, batches, repetitions: int) -> tuple[list[np.ndarray], float]:
    outputs: list[np.ndarray] = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for repetition in range(repetitions):
        current = []
        for batch in batches:
            current.append(runtime.predict_batch(*batch))
        if repetition == repetitions - 1:
            outputs = current
    torch.cuda.synchronize()
    return outputs, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = selected_paths(args.windows)
    batches = [
        load_batch(paths[index : index + args.batch_size])
        for index in range(0, len(paths), args.batch_size)
    ]
    serial = Track2V295TerminalFramePreservingMirror(
        args.checkpoint_dir, args.library_index, args.device
    )
    batched = Track2V301BatchedTerminalFrameMirror(
        args.checkpoint_dir, args.library_index, args.device
    )
    serial_outputs, serial_seconds = timed(serial, batches, args.repetitions)
    batched_outputs, batched_seconds = timed(batched, batches, args.repetitions)
    serial_all = np.concatenate(serial_outputs)
    batched_all = np.concatenate(batched_outputs)
    difference = np.abs(serial_all.astype(np.int16) - batched_all.astype(np.int16))
    report = {
        "format": "strict-track2-v302-v301-batch-equivalence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "windows": len(paths),
        "batch_size": args.batch_size,
        "repetitions": args.repetitions,
        "mismatched_pixels": int(np.count_nonzero(difference)),
        "max_absolute_pixel_change": int(difference.max()),
        "mean_absolute_pixel_change": float(difference.mean()),
        "serial_seconds": serial_seconds,
        "batched_seconds": batched_seconds,
        "speedup": serial_seconds / batched_seconds,
        "passed": bool(np.count_nonzero(difference) == 0),
        "guards": {
            "public_windows_only": True,
            "policy_modified": False,
            "reward_or_outcome_used": False,
            "hidden_or_final_data": False,
            "real_submission": False
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
