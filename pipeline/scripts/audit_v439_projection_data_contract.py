#!/usr/bin/env python3
"""Static, data-boundary, and synthetic-math audit for the v439 core."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v439_v169_action_causal_projection_runtime import (
    ALPHA,
    MU,
    apply_projection,
    clipped_rgb_delta,
    gate_decision,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-index", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.projection_index.read_text())
    split = json.loads(args.split.read_text())
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    source = args.runtime.read_text()
    tree = ast.parse(source, filename=str(args.runtime))
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ]
    called_attributes = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    history = np.zeros((16, 14), dtype=np.float32)
    future = np.zeros((8, 14), dtype=np.float32)
    history[:, 13] = 0.25
    future[:, 13] = 0.25
    future[:, 7:13] = np.linspace(0.0, 1.0, 8)[:, None] * MU[None]
    enabled = gate_decision(history, future, "Use the right arm to adjust the bottle.")
    closing_history = history.copy(); closing_history[:, 13] = 1.0
    closing_future = future.copy(); closing_future[:2, 13] = 1.0
    closing_gate = gate_decision(closing_history, closing_future, "Use the right arm to adjust the bottle.")
    left_prompt = gate_decision(history, future, "Use the left arm to adjust the bottle.")
    open_future = future.copy(); open_future[:, 13] = 1.0
    open_gate = gate_decision(history, open_future, "Use the right arm to adjust the bottle.")
    negative_future = future.copy(); negative_future[:, 7:13] *= -1.0
    negative_gate = gate_decision(history, negative_future, "Use the right arm to adjust the bottle.")

    baseline = np.arange(8 * 4 * 5 * 3, dtype=np.uint16).reshape(8, 4, 5, 3)
    baseline = (baseline % 256).astype(np.uint8)
    parent = np.full_like(baseline, 100)
    candidate = parent.copy()
    candidate[..., 0] = 108
    candidate[..., 1] = 92
    candidate[..., 2] = 104
    delta = clipped_rgb_delta(candidate, parent)
    disabled_output = apply_projection(baseline, candidate, parent, False)
    enabled_output = apply_projection(baseline, candidate, parent, True)
    zero_frames = np.flatnonzero(ALPHA == 0.0)

    checks = {
        "index_format": index.get("format") == "strict-track2-v439-public-right-action-projection-index-v1",
        "split_format": split.get("format") == "strict-track2-v205-public-demo-40train-10holdout-v1",
        "index_train40_exact": index.get("train_episodes") == train and len(train) == 40,
        "validation10_excluded": index.get("validation_episodes_excluded") == validation and len(validation) == 10,
        "train_validation_disjoint": not bool(set(train) & set(validation)),
        "right_train15": len(index.get("right_train_episodes", [])) == 15,
        "no_reward_or_outcome_index": index.get("guards", {}).get("reward_read") is False and index.get("guards", {}).get("outcome_read") is False,
        "runtime_no_reward_import": not any("reward" in name.lower() for name in imports),
        "runtime_no_compute_reward_call": "compute_reward" not in called_attributes,
        "gate_signature_excludes_context_seed_request_id": "def gate_decision(history_actions: np.ndarray, future_actions: np.ndarray, instruction:" in source,
        "frozen_mu": index.get("projection", {}).get("mu_right6d") == MU.astype(float).tolist(),
        "frozen_alpha": index.get("projection", {}).get("alpha_8") == ALPHA.astype(float).tolist(),
        "positive_gate_fixture": enabled["gate"] is True,
        "future_close_then_hold_gate_fixture": closing_gate["gate"] is True,
        "left_prompt_fallback_fixture": left_prompt["gate"] is False,
        "open_gripper_fallback_fixture": open_gate["gate"] is False,
        "negative_projection_fallback_fixture": negative_gate["gate"] is False,
        "signed_rgb_delta_shape": delta.shape == baseline.shape,
        "signed_rgb_delta_bound": int(np.abs(delta).max()) <= 8,
        "g0_all_frames_bitexact": np.array_equal(disabled_output, baseline),
        "g1_alpha_zero_frames_bitexact": np.array_equal(enabled_output[zero_frames], baseline[zero_frames]),
        "candidate_uint8_rgb": enabled_output.dtype == np.uint8 and enabled_output.shape == baseline.shape,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v439-projection-data-contract-v1",
        "passed": passed,
        "checks": checks,
        "synthetic_enabled_decision": enabled,
        "sha256": {
            "projection_index": sha256(args.projection_index),
            "split": sha256(args.split),
            "runtime": sha256(args.runtime),
        },
        "guards": {
            "training_started": False,
            "service_started": False,
            "reward_read": False,
            "outcome_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
