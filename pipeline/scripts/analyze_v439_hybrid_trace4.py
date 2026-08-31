#!/usr/bin/env python3
"""Paired v169-versus-hybrid signal gate over one fresh four-trajectory trace."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PREREG_FORMAT = "strict-track2-v439-hybrid-trace4-preregistration-v1"
TRAJECTORIES = 4
STEPS = 25
REQUESTS = TRAJECTORIES * STEPS
GROUP_SIZE = 4
GAMMA = 0.99
REWARD_COEFFICIENT = 5.0


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


def trajectory_returns(reward: np.ndarray) -> np.ndarray:
    reward_by_time = reward.reshape(STEPS, TRAJECTORIES)
    increments = REWARD_COEFFICIENT * np.diff(
        np.concatenate(
            (np.zeros((1, TRAJECTORIES), dtype=np.float64), reward_by_time),
            axis=0,
        ),
        axis=0,
    )
    discount = np.power(GAMMA, np.arange(STEPS, dtype=np.float64))[:, None]
    return (discount * increments).sum(axis=0)


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
        preregistration.get("format") != PREREG_FORMAT
        or preregistration.get("classification")
        != "parent world-model diagnostic only"
        or preregistration.get("formal_candidate_authorized") is not False
    ):
        raise RuntimeError("unexpected v439 preregistration")

    route_lines = (args.trace_dir / "route_trace.jsonl").read_text().splitlines()
    batches = [json.loads(line)["batch"] for line in route_lines]
    baseline_frames = []
    hybrid_frames = []
    prompts = []
    rows = []
    for index, batch_rows in enumerate(batches):
        stem = f"batch_{index:05d}"
        baseline = np.load(
            args.trace_dir / f"{stem}_v169.npy", allow_pickle=False
        )
        hybrid = np.load(
            args.trace_dir / f"{stem}_hybrid.npy", allow_pickle=False
        )
        batch_prompts = json.loads(
            (args.trace_dir / f"{stem}_prompts.json").read_text()
        )
        if not (
            len(batch_rows) == len(baseline) == len(hybrid) == len(batch_prompts)
        ):
            raise RuntimeError(f"misaligned v439 trace batch {index}")
        if not all(isinstance(value, str) and value for value in batch_prompts):
            raise RuntimeError(f"invalid prompt in v439 trace batch {index}")
        rows.extend(batch_rows)
        baseline_frames.append(baseline)
        hybrid_frames.append(hybrid)
        prompts.extend(batch_prompts)

    baseline_frames_array = np.concatenate(baseline_frames)
    hybrid_frames_array = np.concatenate(hybrid_frames)
    expected_shape = (REQUESTS, 256, 256, 3)
    if (
        len(rows) != REQUESTS
        or baseline_frames_array.shape != expected_shape
        or hybrid_frames_array.shape != expected_shape
        or len(prompts) != REQUESTS
    ):
        raise RuntimeError("v439 requires the intact 4x25/100-request paired trace")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    baseline_reward = score(
        reward_model, baseline_frames_array, prompts, device, args.batch_size
    )
    hybrid_reward = score(
        reward_model, hybrid_frames_array, prompts, device, args.batch_size
    )
    baseline_return = trajectory_returns(baseline_reward)
    hybrid_return = trajectory_returns(hybrid_reward)
    delta_request = hybrid_reward - baseline_reward
    delta_trajectory = hybrid_return - baseline_return
    baseline_spread = float(np.ptp(baseline_return))
    hybrid_spread = float(np.ptp(hybrid_return))
    decision = preregistration["decision"]
    required_hybrid_spread = max(
        float(decision["hybrid_absolute_group_spread_floor"]),
        float(decision["hybrid_absolute_group_spread_ratio_to_v169_min"])
        * baseline_spread,
    )

    checks = {
        "exact_requests_100": len(rows) == REQUESTS,
        "exact_trajectories_4": len(hybrid_return) == TRAJECTORIES,
        "all_request_rewards_finite": bool(
            np.isfinite(baseline_reward).all() and np.isfinite(hybrid_reward).all()
        ),
        "all_trajectory_returns_finite": bool(
            np.isfinite(baseline_return).all() and np.isfinite(hybrid_return).all()
        ),
        "hybrid_absolute_mean_ge_0p90_v169": float(hybrid_return.mean())
        >= float(decision["hybrid_absolute_mean_ratio_to_v169_min"])
        * float(baseline_return.mean()),
        "hybrid_spread_ge_max_0p005_or_half_v169": hybrid_spread
        >= required_hybrid_spread,
        "delta_request_positive_fraction_ge_0p60": float((delta_request > 0).mean())
        >= float(decision["delta_request_positive_fraction_min"]),
        "delta_trajectory_positive_ge_3_of_4": int((delta_trajectory > 0).sum())
        >= int(decision["delta_trajectory_positive_count_min"]),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v439-hybrid-trace4-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "parent world-model diagnostic only",
        "passed": passed,
        "formal_candidate_authorized": False,
        "protocol": {
            "actor_seed": int(preregistration["protocol"]["actor_seed"]),
            "env_seed": int(preregistration["protocol"]["env_seed"]),
            "trajectories": TRAJECTORIES,
            "requests": REQUESTS,
            "steps_per_trajectory": STEPS,
            "group_size": GROUP_SIZE,
            "trajectory_discount": GAMMA,
            "reward_coefficient": REWARD_COEFFICIENT,
            "policy_updates": 0,
            "checkpoint_writes": 0,
        },
        "v169_absolute": {
            "return": baseline_return.tolist(),
            "mean": float(baseline_return.mean()),
            "variance_population": float(baseline_return.var(ddof=0)),
            "positive_fraction": float((baseline_return > 0).mean()),
            "group_spread": baseline_spread,
        },
        "hybrid_absolute": {
            "return": hybrid_return.tolist(),
            "mean": float(hybrid_return.mean()),
            "variance_population": float(hybrid_return.var(ddof=0)),
            "positive_fraction": float((hybrid_return > 0).mean()),
            "group_spread": hybrid_spread,
            "required_group_spread": required_hybrid_spread,
        },
        "delta": {
            "request_mean": float(delta_request.mean()),
            "request_positive_count": int((delta_request > 0).sum()),
            "request_positive_fraction": float((delta_request > 0).mean()),
            "trajectory": delta_trajectory.tolist(),
            "trajectory_positive_count": int((delta_trajectory > 0).sum()),
            "trajectory_positive_fraction": float((delta_trajectory > 0).mean()),
        },
        "checks": checks,
        "guards": {
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
