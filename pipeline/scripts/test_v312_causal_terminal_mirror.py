#!/usr/bin/env python3
"""Deterministic contract tests for v312 before reward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from train_v311_public_action_causal_gate import features as training_features
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v312_causal_terminal_mirror_runtime import (
    Track2V312CausalTerminalMirror,
    action_features,
)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["context_frames"], dtype=np.uint8),
            np.asarray(data["history_actions"], dtype=np.float32),
            np.asarray(data["future_actions"], dtype=np.float32),
        )


def seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--library-index", required=True, type=Path)
    parser.add_argument("--action-gate", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    paths = [
        sorted(args.windows.glob("episode5_*.npz"))[20],
        sorted(args.windows.glob("episode7_*.npz"))[60],
        sorted(args.windows.glob("episode16_*.npz"))[30],
        sorted(args.windows.glob("episode18_*.npz"))[80],
    ]
    samples = [load(path) for path in paths]
    feature_difference = max(
        float(np.max(np.abs(training_features(history, future) - action_features(history, future))))
        for _, history, future in samples
    )
    baseline = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    candidate = Track2V312CausalTerminalMirror(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )
    prompts = ["adjust bottle"] * len(samples)
    seeds = np.asarray([seed(path) for path in paths], dtype=np.int64)
    serial = np.stack(
        [
            candidate.predict(context, history, future, int(value), prompt)
            for (context, history, future), value, prompt in zip(samples, seeds, prompts, strict=True)
        ]
    )
    batch = candidate.predict_batch(
        np.stack([sample[0] for sample in samples]),
        np.stack([sample[1] for sample in samples]),
        np.stack([sample[2] for sample in samples]),
        seeds,
        prompts,
    )
    left_differences = []
    for index in (0, 2):
        context, history, future = samples[index]
        parent = baseline.predict(context, history, future, int(seeds[index]), prompts[index])
        child = candidate.predict(context, history, future, int(seeds[index]), prompts[index])
        left_differences.append(int(np.abs(parent.astype(np.int16) - child.astype(np.int16)).max()))
    checks = {
        "training_runtime_features_bit_exact": feature_difference == 0.0,
        "serial_batch_bit_exact": bool(np.array_equal(serial, batch)),
        "left_parent_bit_exact": max(left_differences) == 0,
        "shape_and_dtype": batch.shape == (4, 8, 256, 256, 3) and batch.dtype == np.uint8,
    }
    report = {
        "format": "strict-track2-v312-causal-terminal-mirror-contract-v1",
        "feature_max_absolute_difference": feature_difference,
        "serial_batch_max_absolute_pixel_difference": int(
            np.abs(serial.astype(np.int16) - batch.astype(np.int16)).max()
        ),
        "left_max_absolute_pixel_differences": left_differences,
        "checks": checks,
        "passed": all(checks.values()),
        "guards": {
            "public_windows_only": True,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
