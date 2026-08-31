#!/usr/bin/env python3
"""Read-only absolute-return forensic over the existing v436 trace.

This report is descriptive only.  It does not replace, edit, or reinterpret
the preregistered v436 pass/fail decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


GAMMA = 0.99
REWARD_COEFFICIENT = 5.0
TRAJECTORIES = 8
REQUESTS_PER_TRAJECTORY = 25
REQUESTS = TRAJECTORIES * REQUESTS_PER_TRAJECTORY
GROUP_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(reward_model, frames: np.ndarray, prompts: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    for begin in range(0, len(frames), batch_size):
        end = min(begin + batch_size, len(frames))
        tensor = (
            torch.from_numpy(np.ascontiguousarray(frames[begin:end]))
            .permute(0, 3, 1, 2)
            .float()
            .div(255)
            .to(device)
        )
        with torch.inference_mode():
            output = reward_model.compute_reward(
                tensor, task_descriptions=prompts[begin:end]
            )
        values.append(output.detach().float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def absolute_returns(scores: np.ndarray) -> np.ndarray:
    """Apply the v436 5*diff reward transform to one score series per trajectory."""
    if scores.shape != (REQUESTS_PER_TRAJECTORY, TRAJECTORIES):
        raise RuntimeError(f"unexpected score matrix: {scores.shape}")
    increments = np.diff(
        np.concatenate(
            (np.zeros((1, TRAJECTORIES), dtype=np.float64), scores), axis=0
        ),
        axis=0,
    )
    discount = np.power(GAMMA, np.arange(REQUESTS_PER_TRAJECTORY))[:, None]
    return (REWARD_COEFFICIENT * discount * increments).sum(axis=0)


def summarize(returns: np.ndarray) -> dict:
    if returns.shape != (TRAJECTORIES,):
        raise RuntimeError(f"unexpected return vector: {returns.shape}")
    group_spreads = [
        float(np.ptp(returns[begin : begin + GROUP_SIZE]))
        for begin in range(0, TRAJECTORIES, GROUP_SIZE)
    ]
    return {
        "returns": returns.tolist(),
        "mean": float(returns.mean()),
        "variance_population": float(returns.var(ddof=0)),
        "positive_count": int((returns > 0).sum()),
        "positive_fraction": float((returns > 0).mean()),
        "group_spreads": group_spreads,
        "group_spread_mean": float(np.mean(group_spreads)),
        "group_spread_max": float(np.max(group_spreads)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "trace-dir",
        "preregistration",
        "source-result",
        "reward-checkpoint",
        "t5-model",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.output.resolve() == args.source_result.resolve():
        raise RuntimeError("forensic output must not overwrite the v436 result")
    if args.output.exists():
        raise FileExistsError(args.output)

    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("format")
        != "strict-track2-v436-v432-step25-trace8-preregistration-v1"
        or preregistration.get("classification")
        != "parent world-model diagnostic only"
        or preregistration.get("formal_candidate_authorized") is not False
    ):
        raise RuntimeError("unexpected v436 preregistration")
    source_result = json.loads(args.source_result.read_text())
    if source_result.get("format") != "strict-track2-v436-v432-step25-trace8-result-v1":
        raise RuntimeError("unexpected v436 source result")

    route_path = args.trace_dir / "route_trace.jsonl"
    batches = [json.loads(line)["batch"] for line in route_path.read_text().splitlines()]
    parent_frames = []
    candidate_frames = []
    prompts: list[str] = []
    rows = []
    for batch_index, batch_rows in enumerate(batches):
        stem = f"batch_{batch_index:05d}"
        parent = np.load(args.trace_dir / f"{stem}_parent.npy", allow_pickle=False)
        candidate = np.load(args.trace_dir / f"{stem}_candidate.npy", allow_pickle=False)
        batch_prompts = json.loads(
            (args.trace_dir / f"{stem}_prompts.json").read_text()
        )
        if not (len(batch_rows) == len(parent) == len(candidate) == len(batch_prompts)):
            raise RuntimeError(f"trace alignment failed at {stem}")
        rows.extend(batch_rows)
        parent_frames.append(parent)
        candidate_frames.append(candidate)
        prompts.extend(batch_prompts)

    parent_frames_array = np.concatenate(parent_frames)
    candidate_frames_array = np.concatenate(candidate_frames)
    expected_shape = (REQUESTS, 256, 256, 3)
    if (
        len(rows) != REQUESTS
        or parent_frames_array.shape != expected_shape
        or candidate_frames_array.shape != expected_shape
        or len(prompts) != REQUESTS
    ):
        raise RuntimeError("v436 forensic requires the intact 8x25/200-request trace")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    parent_scores = score(
        reward_model, parent_frames_array, prompts, device, args.batch_size
    ).reshape(REQUESTS_PER_TRAJECTORY, TRAJECTORIES)
    candidate_scores = score(
        reward_model, candidate_frames_array, prompts, device, args.batch_size
    ).reshape(REQUESTS_PER_TRAJECTORY, TRAJECTORIES)

    parent_return = absolute_returns(parent_scores)
    candidate_return = absolute_returns(candidate_scores)
    recomputed_delta = candidate_return - parent_return
    source_delta = np.asarray(
        source_result.get("trajectory", {}).get("return_delta", []), dtype=np.float64
    )
    if source_delta.shape != (TRAJECTORIES,):
        raise RuntimeError("source v436 result lacks the eight delta returns")
    delta_error = np.abs(recomputed_delta - source_delta)

    report = {
        "format": "strict-track2-v437-v436-absolute-return-forensic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "read-only parent world-model forensic diagnostic",
        "formal_candidate_authorized": False,
        "method": {
            "official_reward_checkpoint": str(args.reward_checkpoint.resolve()),
            "gamma": GAMMA,
            "reward_coefficient": REWARD_COEFFICIENT,
            "transform": "5 * diff([0, reward_score_t]), discounted by gamma**t",
            "trajectories": TRAJECTORIES,
            "requests": REQUESTS,
            "requests_per_trajectory": REQUESTS_PER_TRAJECTORY,
            "group_size": GROUP_SIZE,
        },
        "parent_absolute": summarize(parent_return),
        "candidate_absolute": summarize(candidate_return),
        "candidate_minus_parent": {
            "returns": recomputed_delta.tolist(),
            "mean": float(recomputed_delta.mean()),
            "variance_population": float(recomputed_delta.var(ddof=0)),
            "positive_count": int((recomputed_delta > 0).sum()),
            "positive_fraction": float((recomputed_delta > 0).mean()),
            "group_spreads": [
                float(np.ptp(recomputed_delta[begin : begin + GROUP_SIZE]))
                for begin in range(0, TRAJECTORIES, GROUP_SIZE)
            ],
        },
        "source_delta_consistency": {
            "source_returns": source_delta.tolist(),
            "max_absolute_error": float(delta_error.max()),
            "matches_at_1e_minus_8": bool(np.all(delta_error <= 1e-8)),
        },
        "evidence_sha256": {
            "route_trace": sha256(route_path),
            "preregistration": sha256(args.preregistration),
            "source_v436_result": sha256(args.source_result),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "source_v436_decision": {
            "passed": source_result.get("passed"),
            "preserved_unchanged": True,
        },
        "guards": {
            "writes_v436_result": False,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
