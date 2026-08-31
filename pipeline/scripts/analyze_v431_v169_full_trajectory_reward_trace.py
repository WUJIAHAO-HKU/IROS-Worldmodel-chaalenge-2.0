#!/usr/bin/env python3
"""Audit raw reward rankability for the original v169 diagnostic world model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


EXPECTED_PREREGISTRATION = (
    "strict-track2-v431-v169-diagnostic-full-trajectory-reward-trace32-preregistration-v1"
)
MODEL_VERSION = "track2-v16.9-instruction-arm-routed"
BACKEND = "v169-arm-routed"


def score(model, frames, prompts, device, batch_size):
    values = []
    for begin in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[begin : begin + batch_size]).permute(0, 3, 1, 2)
        with torch.inference_mode():
            result = model.compute_reward(
                batch.float().div(255.0).to(device),
                prompts[begin : begin + batch_size],
            )
        values.append(result.float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("trace-dir", "preregistration", "reward-checkpoint", "t5-model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != EXPECTED_PREREGISTRATION:
        raise RuntimeError("wrong v431 preregistration")
    if prereg.get("classification") != "nonformal diagnostic only":
        raise RuntimeError("v431 must remain nonformal diagnostic")
    if args.output.exists():
        raise FileExistsError(args.output)

    route_lines = (args.trace_dir / "route_trace.jsonl").read_text().splitlines()
    batches = [json.loads(line)["batch"] for line in route_lines]
    frames, prompts = [], []
    for index, routes in enumerate(batches):
        stem = f"batch_{index:05d}"
        batch_frames = np.load(args.trace_dir / f"{stem}_v169.npy", allow_pickle=False)
        texts = json.loads((args.trace_dir / f"{stem}_prompts.json").read_text())
        if len(routes) != len(texts) or len(routes) != len(batch_frames):
            raise RuntimeError(f"misaligned trace batch {index}")
        if not all(isinstance(text, str) and text for text in texts):
            raise RuntimeError(f"invalid prompt in trace batch {index}")
        frames.append(batch_frames)
        prompts.extend(texts)
    rows = [row for batch in batches for row in batch]
    frames = np.concatenate(frames)
    if len(rows) != 800 or frames.shape != (800, 256, 256, 3):
        raise RuntimeError("expected exact 800-request v169 trace")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    reward = score(
        reward_model, frames, prompts, torch.device(args.device), args.batch_size
    )

    # Native microbatches are time-major: four batches of eight requests form
    # one 32-trajectory step, repeated for the exact 25 world-model steps.
    reward_by_time = reward.reshape(25, 32)
    relative_reward = 5.0 * np.diff(
        np.concatenate([np.zeros((1, 32), dtype=np.float64), reward_by_time], axis=0),
        axis=0,
    )
    gamma = float(prereg["decision"]["trajectory_discount"])
    discount = np.power(gamma, np.arange(25, dtype=np.float64))[:, None]
    trajectory_return = (discount * relative_reward).sum(axis=0)
    group_size = int(prereg["protocol"]["group_size"])
    group_return_spread = [
        float(np.ptp(trajectory_return[begin : begin + group_size]))
        for begin in range(0, len(trajectory_return), group_size)
    ]

    decision = prereg["decision"]
    right_requests = sum(row["arm"] == "right" for row in rows)
    groups_with_spread = sum(
        value >= float(decision["group_return_spread_min"])
        for value in group_return_spread
    )
    checks = {
        "exact_requests": len(rows) == 800,
        "exact_trajectories": len(trajectory_return) == 32,
        "all_request_rewards_finite": bool(np.isfinite(reward).all()),
        "all_trajectory_returns_finite": bool(np.isfinite(trajectory_return).all()),
        "mean_trajectory_return_ge_0p06": float(trajectory_return.mean())
        >= float(decision["mean_trajectory_return_min"]),
        "trajectory_return_variance_nonzero": float(trajectory_return.std()) > 1e-12,
        "at_least_four_groups_have_return_spread_ge_0p02": groups_with_spread
        >= int(decision["groups_with_spread_min"]),
        "right_routed_requests_ge_200": right_requests
        >= int(decision["right_routed_requests_min"]),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v430-v169-full-trajectory-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "nonformal diagnostic only",
        "passed": passed,
        "formal_candidate_authorized": False,
        "world_model": {"model_version": MODEL_VERSION, "backend": BACKEND},
        "protocol": {
            "actor_seed": int(prereg["protocol"]["actor_seed"]),
            "env_seed": int(prereg["protocol"]["env_seed"]),
            "trajectories": 32,
            "requests": 800,
            "steps_per_trajectory": 25,
            "group_size": group_size,
            "episode_steps": 200,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "official_relative_reward": True,
            "trajectory_discount": gamma,
        },
        "requests": 800,
        "trajectories": 32,
        "route": {
            "right_requests": right_requests,
            "left_requests": len(rows) - right_requests,
            "explicit_instruction_requests": sum(
                row["explicit_instruction_route"] for row in rows
            ),
        },
        "reward": {
            "request_mean": float(reward.mean()),
            "request_std": float(reward.std()),
            "request_min": float(reward.min()),
            "request_max": float(reward.max()),
            "hits_ge_0p9": int((reward >= 0.9).sum()),
            "final_mean": float(reward_by_time[-1].mean()),
            "positive_relative_reward_fraction": float((relative_reward > 0).mean()),
        },
        "trajectory": {
            "discount": gamma,
            "mean_return": float(trajectory_return.mean()),
            "median_return": float(np.median(trajectory_return)),
            "return_std": float(trajectory_return.std()),
            "positive_return_fraction": float((trajectory_return > 0).mean()),
            "return": trajectory_return.tolist(),
            "group_return_spread": group_return_spread,
            "groups_with_spread": groups_with_spread,
        },
        "checks": checks,
        "guards": {
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "reward_is_official_training_signal_not_simulator_outcome": True,
            "hidden_or_final_data": False,
            "real_submission": False,
            "diagnostic_cannot_select_formal_candidate": True,
        },
        "rows": [
            {**row, "reward": float(reward[index])}
            for index, row in enumerate(rows)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
