#!/usr/bin/env python3
"""Numeric, speed, and branch contract for native-batched v317."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from audit_v310_full_mirror_causal_gate import make_counterfactual
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import Track2V271EndpointCalibratedTerminal
from wam_pipeline.v312_causal_terminal_mirror_runtime import action_features
from wam_pipeline.v315_sparse_failure_terminal_runtime import Track2V315SparseFailureTerminal
from wam_pipeline.v317_batched_sparse_failure_terminal_runtime import Track2V317BatchedSparseFailureTerminal
from train_v311_public_action_causal_gate import features as training_features


def load(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return tuple(
            np.asarray(data[name], dtype=dtype)
            for name, dtype in (
                ("context_frames", np.uint8),
                ("history_actions", np.float32),
                ("future_actions", np.float32),
            )
        )


def seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


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

    paths = []
    for episode in (5, 7, 16, 18):
        episode_paths = sorted(args.windows.glob(f"episode{episode}_*.npz"))
        paths.extend(episode_paths[index] for index in (20, 60))
    samples = [load(path) for path in paths]
    contexts = np.stack([sample[0] for sample in samples])
    histories = np.stack([sample[1] for sample in samples])
    futures = np.stack([sample[2] for sample in samples])
    seeds = np.asarray([seed(path) for path in paths], dtype=np.int64)
    prompts = ["adjust bottle"] * len(samples)
    serial_runtime = Track2V315SparseFailureTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )
    batch_runtime = Track2V317BatchedSparseFailureTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )
    baseline = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )

    serial = np.stack([
        serial_runtime.predict(context, history, future, int(value), prompt)
        for (context, history, future), value, prompt in zip(samples, seeds, prompts, strict=True)
    ])
    batch = batch_runtime.predict_batch(contexts, histories, futures, seeds, prompts)
    difference = np.abs(serial.astype(np.int16) - batch.astype(np.int16))

    # Warm both paths before timing and use medians to avoid one-time compilation noise.
    batch_runtime.predict_batch(contexts, histories, futures, seeds, prompts)
    serial_times = []
    batch_times = []
    for _ in range(3):
        synchronize(args.device)
        begin = time.perf_counter()
        np.stack([
            serial_runtime.predict(context, history, future, int(value), prompt)
            for (context, history, future), value, prompt in zip(samples, seeds, prompts, strict=True)
        ])
        synchronize(args.device)
        serial_times.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        batch_runtime.predict_batch(contexts, histories, futures, seeds, prompts)
        synchronize(args.device)
        batch_times.append(time.perf_counter() - begin)

    left_single_differences = []
    for index in (0, 1, 4, 5):
        context, history, future = samples[index]
        expected = baseline.predict(context, history, future, int(seeds[index]), prompts[index])
        actual = batch_runtime.predict(context, history, future, int(seeds[index]), prompts[index])
        left_single_differences.append(int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()))

    transition_path = args.windows / "episode7_00098.npz"
    context, history, future = load(transition_path)
    variants = [future] + [
        make_counterfactual(future, history, name)
        for name in ("open_gripper", "static_transport", "reverse_transport")
    ]
    branch = batch_runtime.predict_batch(
        np.repeat(context[None], 4, axis=0),
        np.repeat(history[None], 4, axis=0),
        np.stack(variants),
        np.repeat(seed(transition_path), 4),
        ["adjust bottle"] * 4,
    )
    direct = baseline.predict(context, history, future, seed(transition_path), "adjust bottle")
    feature_difference = max(
        float(np.max(np.abs(training_features(history, future) - action_features(history, future))))
        for _, history, future in samples
    )
    serial_median = float(np.median(serial_times))
    batch_median = float(np.median(batch_times))
    checks = {
        "training_runtime_features_bit_exact": feature_difference == 0.0,
        "batch_max_absolute_pixel_change_le_2": int(difference.max()) <= 2,
        "batch_mean_absolute_pixel_change_le_0p05": float(difference.mean()) <= 0.05,
        "native_batch_speedup_ge_1p5": serial_median / batch_median >= 1.5,
        "left_single_parent_bit_exact": max(left_single_differences) == 0,
        "valid_transition_terminal_numeric_within_2": int(
            np.abs(branch[0, -1].astype(np.int16) - direct[-1].astype(np.int16)).max()
        ) <= 2,
        "all_counterfactual_terminals_context_exact": all(
            np.array_equal(branch[index, -1], context[-1]) for index in (1, 2, 3)
        ),
        "shape_and_dtype": batch.shape == (8, 8, 256, 256, 3) and batch.dtype == np.uint8,
    }
    report = {
        "format": "strict-track2-v317-batched-sparse-failure-contract-v1",
        "candidate": "track2-v317-batched-sparse-failure-terminal-v315",
        "feature_max_absolute_difference": feature_difference,
        "serial_batch": {
            "max_absolute_pixel_change": int(difference.max()),
            "mean_absolute_pixel_change": float(difference.mean()),
            "serial_seconds_median": serial_median,
            "batch_seconds_median": batch_median,
            "speedup": serial_median / batch_median,
        },
        "left_single_max_absolute_pixel_differences": left_single_differences,
        "checks": checks,
        "passed": all(checks.values()),
        "guards": {
            "public_windows_only": True,
            "runtime_reads_reward_or_outcome": False,
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
