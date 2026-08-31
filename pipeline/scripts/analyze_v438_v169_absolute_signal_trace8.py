#!/usr/bin/env python3
"""Gate the absolute official-reward signal of a fresh v169 trace8 rollout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


FORMAT = "strict-track2-v438-v169-absolute-signal-trace8-preregistration-v1"
TRAJECTORIES = 8
STEPS = 25
REQUESTS = TRAJECTORIES * STEPS


def score(model, frames, prompts, device, batch_size):
    values = []
    for begin in range(0, len(frames), batch_size):
        tensor = torch.from_numpy(
            np.ascontiguousarray(frames[begin : begin + batch_size])
        ).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.inference_mode():
            output = model.compute_reward(
                tensor, task_descriptions=prompts[begin : begin + batch_size]
            )
        values.append(output.detach().float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "trace-dir",
        "preregistration",
        "reward-checkpoint",
        "t5-model",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("format") != FORMAT
        or preregistration.get("classification") != "nonformal diagnostic only"
        or preregistration.get("formal_candidate_authorized") is not False
    ):
        raise RuntimeError("unexpected v438 preregistration")

    route_lines = (args.trace_dir / "route_trace.jsonl").read_text().splitlines()
    batches = [json.loads(line)["batch"] for line in route_lines]
    frames = []
    prompts = []
    rows = []
    for index, routes in enumerate(batches):
        stem = f"batch_{index:05d}"
        batch_frames = np.load(
            args.trace_dir / f"{stem}_v169.npy", allow_pickle=False
        )
        batch_prompts = json.loads(
            (args.trace_dir / f"{stem}_prompts.json").read_text()
        )
        if not (len(routes) == len(batch_frames) == len(batch_prompts)):
            raise RuntimeError(f"misaligned v438 trace batch {index}")
        if not all(isinstance(value, str) and value for value in batch_prompts):
            raise RuntimeError(f"invalid prompt in v438 trace batch {index}")
        rows.extend(routes)
        frames.append(batch_frames)
        prompts.extend(batch_prompts)

    frames_array = np.concatenate(frames)
    if len(rows) != REQUESTS or frames_array.shape != (REQUESTS, 256, 256, 3):
        raise RuntimeError("v438 requires exactly 8 trajectories and 200 requests")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    reward = score(
        reward_model, frames_array, prompts, device, args.batch_size
    )
    reward_by_time = reward.reshape(STEPS, TRAJECTORIES)

    decision = preregistration["decision"]
    gamma = float(decision["trajectory_discount"])
    relative_reward = 5.0 * np.diff(
        np.concatenate(
            (np.zeros((1, TRAJECTORIES), dtype=np.float64), reward_by_time),
            axis=0,
        ),
        axis=0,
    )
    discount = np.power(gamma, np.arange(STEPS, dtype=np.float64))[:, None]
    trajectory_return = (discount * relative_reward).sum(axis=0)
    group_size = int(preregistration["protocol"]["group_size"])
    group_spread = [
        float(np.ptp(trajectory_return[begin : begin + group_size]))
        for begin in range(0, TRAJECTORIES, group_size)
    ]
    groups_with_spread = sum(
        value >= float(decision["absolute_group_spread_min"])
        for value in group_spread
    )

    checks = {
        "exact_requests_200": len(rows) == REQUESTS,
        "exact_trajectories_8": len(trajectory_return) == TRAJECTORIES,
        "all_request_rewards_finite": bool(np.isfinite(reward).all()),
        "all_trajectory_returns_finite": bool(np.isfinite(trajectory_return).all()),
        "mean_absolute_return_ge_0p06": float(trajectory_return.mean())
        >= float(decision["mean_absolute_return_min"]),
        "at_least_one_group_absolute_spread_ge_0p02": groups_with_spread
        >= int(decision["groups_with_spread_min"]),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v438-v169-absolute-signal-trace8-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "nonformal diagnostic only",
        "passed": passed,
        "formal_candidate_authorized": False,
        "world_model": {
            "model_version": "track2-v16.9-instruction-arm-routed",
            "backend": "v431-v169-reward-trace",
            "candidate": "original frozen v169 world model",
        },
        "protocol": {
            "actor_seed": int(preregistration["protocol"]["actor_seed"]),
            "env_seed": int(preregistration["protocol"]["env_seed"]),
            "trajectories": TRAJECTORIES,
            "requests": REQUESTS,
            "steps_per_trajectory": STEPS,
            "group_size": group_size,
            "trajectory_discount": gamma,
            "reward_coefficient": 5.0,
            "policy_updates": 0,
            "checkpoint_writes": 0,
        },
        "candidate_absolute": {
            "return": trajectory_return.tolist(),
            "mean_return": float(trajectory_return.mean()),
            "variance_population": float(trajectory_return.var(ddof=0)),
            "positive_count": int((trajectory_return > 0).sum()),
            "positive_fraction": float((trajectory_return > 0).mean()),
            "group_spread": group_spread,
            "groups_with_spread_ge_0p02": groups_with_spread,
        },
        "reward": {
            "request_mean": float(reward.mean()),
            "request_std": float(reward.std()),
            "request_min": float(reward.min()),
            "request_max": float(reward.max()),
            "final_mean": float(reward_by_time[-1].mean()),
            "positive_relative_reward_fraction": float((relative_reward > 0).mean()),
        },
        "route": {
            "right_requests": sum(row["arm"] == "right" for row in rows),
            "left_requests": sum(row["arm"] == "left" for row in rows),
        },
        "checks": checks,
        "guards": {
            "opponent_or_delta_required": False,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
            "diagnostic_cannot_select_formal_candidate": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
