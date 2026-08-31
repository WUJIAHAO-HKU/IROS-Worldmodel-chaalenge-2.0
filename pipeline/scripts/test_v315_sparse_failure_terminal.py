#!/usr/bin/env python3
"""Deterministic contract and branch tests for v315."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_v310_full_mirror_causal_gate import make_counterfactual
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v295_terminal_frame_preserving_mirror_runtime import (
    Track2V295TerminalFramePreservingMirror,
)
from wam_pipeline.v315_sparse_failure_terminal_runtime import (
    Track2V315SparseFailureTerminal,
)
from wam_pipeline.v312_causal_terminal_mirror_runtime import action_features
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
    prompts = ["adjust bottle"] * len(samples)
    seeds = np.asarray([seed(path) for path in paths], dtype=np.int64)
    candidate = Track2V315SparseFailureTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )
    baseline = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    parent = Track2V295TerminalFramePreservingMirror(
        args.checkpoint_dir, args.library_index, args.device
    )
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
    feature_difference = max(
        float(np.max(np.abs(training_features(history, future) - action_features(history, future))))
        for _, history, future in samples
    )
    left_differences = []
    for index in (0, 2):
        context, history, future = samples[index]
        expected = baseline.predict(context, history, future, int(seeds[index]), prompts[index])
        actual = candidate.predict(context, history, future, int(seeds[index]), prompts[index])
        left_differences.append(int(np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max()))

    transition_path = args.windows / "episode7_00098.npz"
    context, history, future = load(transition_path)
    transition_seed = seed(transition_path)
    valid = candidate.predict(context, history, future, transition_seed, "adjust bottle")
    valid_suppressed = candidate.last_failure_suppressed
    direct = baseline.predict(context, history, future, transition_seed, "adjust bottle")
    failure_checks = {}
    signatures = {}
    for name in ("open_gripper", "static_transport", "reverse_transport"):
        altered = make_counterfactual(future, history, name)
        predicted = candidate.predict(context, history, altered, transition_seed, "adjust bottle")
        failure_checks[name] = bool(np.array_equal(predicted[-1], context[-1]))
        signatures[name] = candidate.last_failure_signature

    normal_path = args.windows / "episode7_00045.npz"
    normal_context, normal_history, normal_future = load(normal_path)
    normal_seed = seed(normal_path)
    actual = candidate.predict(
        normal_context, normal_history, normal_future, normal_seed, "adjust bottle"
    )
    expected = parent.predict(
        normal_context, normal_history, normal_future, normal_seed, "adjust bottle"
    )
    normal_terminal_is_context = bool(np.array_equal(actual[-1], normal_context[-1]))
    normal_nonterminal_parent_exact = bool(np.array_equal(actual[:-1], expected[:-1]))

    checks = {
        "training_runtime_features_bit_exact": feature_difference == 0.0,
        "serial_batch_bit_exact": bool(np.array_equal(serial, batch)),
        "left_parent_bit_exact": max(left_differences) == 0,
        "valid_transition_not_suppressed": not valid_suppressed,
        "valid_transition_terminal_direct_exact": bool(np.array_equal(valid[-1], direct[-1])),
        "all_counterfactual_terminals_context_exact": all(failure_checks.values()),
        "normal_failure_nonterminal_v295_exact": normal_nonterminal_parent_exact,
        "normal_failure_terminal_context_exact": normal_terminal_is_context,
        "shape_and_dtype": batch.shape == (4, 8, 256, 256, 3) and batch.dtype == np.uint8,
    }
    report = {
        "format": "strict-track2-v315-sparse-failure-terminal-contract-v1",
        "candidate": "track2-v315-sparse-failure-terminal-v271",
        "feature_max_absolute_difference": feature_difference,
        "serial_batch_max_absolute_pixel_difference": int(
            np.abs(serial.astype(np.int16) - batch.astype(np.int16)).max()
        ),
        "left_max_absolute_pixel_differences": left_differences,
        "counterfactual_signatures": signatures,
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
