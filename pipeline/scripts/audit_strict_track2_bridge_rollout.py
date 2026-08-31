#!/usr/bin/env python3
"""Verify an audited official RL chunk is replayable and routed as declared."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v15_gated_runtime import Track2V15GatedRuntime


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expected-route",
        choices=("candidate", "baseline", "any"),
        default="candidate",
    )
    args = parser.parse_args()
    audit_path = Path(args.audit).resolve()
    with np.load(audit_path, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    context = arrays["context_frames"]
    history = arrays["history_actions"]
    future = arrays["future_actions"]
    prediction = arrays["predicted_frames"]
    seeds = arrays["seeds"]
    instructions = json.loads(str(arrays["instructions_json"]))
    batch = len(context)
    if context.shape != (batch, 5, 256, 256, 3):
        raise ValueError(f"unexpected context shape {context.shape}")
    if history.shape != (batch, 4, 14) or future.shape != (batch, 8, 14):
        raise ValueError("unexpected action shapes")
    if prediction.shape != (batch, 8, 256, 256, 3):
        raise ValueError(f"unexpected prediction shape {prediction.shape}")
    runtime = Track2V15GatedRuntime(args.release, args.library, args.device)
    rows = []
    for index in range(batch):
        replay = runtime.predict(
            context[index], history[index], future[index], int(seeds[index]), instructions[index]
        )
        rows.append(
            {
                "batch_index": index,
                "seed": int(seeds[index]),
                "instruction": instructions[index],
                "route": runtime.gated_autoregressive.last_route,
                "probability_onpolicy": runtime.gated_autoregressive.last_probability_synthetic,
                "active_arm": runtime.gated_autoregressive.last_arm_route,
                "future_actions_sha256": array_sha256(future[index]),
                "prediction_sha256": array_sha256(prediction[index]),
                "replay_bit_exact": bool(np.array_equal(replay, prediction[index])),
            }
        )
    report = {
        "format": "strict-track2-bridge-rollout-audit-v1",
        "audit": str(audit_path),
        "batch_size": batch,
        "context_shape": list(context.shape),
        "history_action_shape": list(history.shape),
        "future_action_shape": list(future.shape),
        "prediction_shape": list(prediction.shape),
        "candidate_routes": sum(str(row["route"]).startswith("candidate") for row in rows),
        "all_replays_bit_exact": all(row["replay_bit_exact"] for row in rows),
        "group_future_action_l2": float(np.linalg.norm(future[0] - future[1])) if batch == 2 else None,
        "future_action_min": float(future.min()),
        "future_action_max": float(future.max()),
        "future_action_mean": float(future.mean()),
        "future_gripper_mean": {
            "left": float(future[:, :, 6].mean()),
            "right": float(future[:, :, 13].mean()),
        },
        "rows": rows,
        "action_integrity": "The bridge stores the same contiguous tensor that is passed unchanged to predict_batch; bridge unit test and runtime hashes are frozen separately.",
        "participant_action_selection": False,
        "mpc": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    route_ok = (
        args.expected_route == "any"
        or (args.expected_route == "candidate" and report["candidate_routes"] == batch)
        or (args.expected_route == "baseline" and report["candidate_routes"] == 0)
    )
    if not route_ok or not report["all_replays_bit_exact"]:
        raise RuntimeError(f"bridge audit failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
