#!/usr/bin/env python3
"""Audit the frozen V15 action router without loading either video expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def probability(gate: dict, history: np.ndarray, future: np.ndarray) -> float:
    actions = np.concatenate((history, future), axis=0).astype(np.float32)
    delta = np.diff(actions, axis=0)
    summary = np.concatenate(
        (
            actions.mean(0),
            actions.std(0),
            np.abs(delta[:, :7]).mean(0),
            np.abs(delta[:, 7:]).mean(0),
            np.asarray(
                [np.abs(delta[:, :7]).mean(), np.abs(delta[:, 7:]).mean()],
                dtype=np.float32,
            ),
        )
    )
    feature = np.concatenate((actions.reshape(-1), delta.reshape(-1), summary))
    value = torch.from_numpy(feature).float().unsqueeze(0)
    normalized = (value - gate["feature_mean"].float()) / gate["feature_std"].float()
    state = gate["state_dict"]
    with torch.inference_mode():
        hidden = torch.nn.functional.silu(
            torch.nn.functional.linear(
                normalized, state["first.weight"], state["first.bias"]
            )
        )
        logit = torch.nn.functional.linear(
            hidden, state["second.weight"], state["second.bias"]
        )
    return float(logit.sigmoid()[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gate = torch.load(args.gate, map_location="cpu", weights_only=True)
    if gate.get("format") != "strict-track2-action-source-gate-v1":
        raise RuntimeError("expected the frozen Track 2 action source gate")
    threshold = float(gate["threshold"])
    with np.load(args.audit, allow_pickle=False) as values:
        history = values["history_actions"].copy()
        future = values["future_actions"].copy()
    if history.shape != (len(history), 4, 14) or future.shape != (len(history), 8, 14):
        raise ValueError("unexpected Track 2 action audit shapes")

    rows = []
    for index in range(len(history)):
        value = probability(gate, history[index], future[index])
        rows.append(
            {
                "batch_index": index,
                "probability_onpolicy": value,
                "threshold": threshold,
                "route": "candidate" if value >= threshold else "baseline",
                "history_sha256": array_sha256(history[index]),
                "future_sha256": array_sha256(future[index]),
            }
        )
    report = {
        "format": "strict-track2-action-route-audit-v1",
        "audit": str(args.audit.resolve()),
        "gate": str(args.gate.resolve()),
        "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
        "batch_size": len(rows),
        "candidate_routes": sum(row["route"] == "candidate" for row in rows),
        "baseline_routes": sum(row["route"] == "baseline" for row in rows),
        "group_future_action_l2": (
            float(np.linalg.norm(future[0] - future[1])) if len(future) == 2 else None
        ),
        "future_action_min": float(future.min()),
        "future_action_max": float(future.max()),
        "future_gripper_mean": {
            "left": float(future[:, :, 6].mean()),
            "right": float(future[:, :, 13].mean()),
        },
        "participant_action_selection": False,
        "mpc": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
