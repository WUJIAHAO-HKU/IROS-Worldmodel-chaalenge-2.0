#!/usr/bin/env python3
"""Score v209/v426 traces and require useful per-request and trajectory signal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def score(model, frames, prompts, device, batch_size):
    values = []
    for begin in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[begin : begin + batch_size]).permute(0, 3, 1, 2)
        with torch.inference_mode():
            result = model.compute_reward(batch.float().div(255.0).to(device), prompts[begin : begin + batch_size])
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
    if prereg.get("format") != "strict-track2-v429-v426-full-trajectory-reward-trace32-preregistration-v1":
        raise RuntimeError("wrong v429 preregistration")
    if args.output.exists():
        raise FileExistsError(args.output)
    batches = [json.loads(line)["batch"] for line in (args.trace_dir / "route_trace.jsonl").read_text().splitlines()]
    baseline_frames, candidate_frames, prompts = [], [], []
    for index, routes in enumerate(batches):
        stem = f"batch_{index:05d}"
        baseline = np.load(args.trace_dir / f"{stem}_v209.npy", allow_pickle=False)
        candidate = np.load(args.trace_dir / f"{stem}_v426.npy", allow_pickle=False)
        texts = json.loads((args.trace_dir / f"{stem}_prompts.json").read_text())
        if baseline.shape != candidate.shape or len(routes) != len(texts) or len(routes) != len(baseline):
            raise RuntimeError(f"misaligned trace batch {index}")
        baseline_frames.append(baseline)
        candidate_frames.append(candidate)
        prompts.extend(texts)
    rows = [row for batch in batches for row in batch]
    baseline_frames, candidate_frames = np.concatenate(baseline_frames), np.concatenate(candidate_frames)
    if len(rows) != 800 or baseline_frames.shape != (800, 256, 256, 3) or candidate_frames.shape != baseline_frames.shape:
        raise RuntimeError("expected exact 800-request trace")
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    baseline_score = score(reward, baseline_frames, prompts, torch.device(args.device), args.batch_size)
    candidate_score = score(reward, candidate_frames, prompts, torch.device(args.device), args.batch_size)
    delta = candidate_score - baseline_score
    learned = np.asarray([index for index, row in enumerate(rows) if row["learned_right_route"]], dtype=np.int64)
    learned_delta = delta[learned]

    # Native microbatches are time-major: four batches of eight requests form
    # one 32-trajectory world-model step, repeated for 25 steps.
    baseline_by_time = baseline_score.reshape(25, 32)
    candidate_by_time = candidate_score.reshape(25, 32)
    delta_by_time = candidate_by_time - baseline_by_time
    incremental_delta = np.diff(np.concatenate([np.zeros((1, 32)), delta_by_time], axis=0), axis=0)
    gamma = float(prereg["decision"]["trajectory_discount"])
    discount = np.power(gamma, np.arange(25, dtype=np.float64))[:, None]
    trajectory_return_delta = (5.0 * discount * incremental_delta).sum(axis=0)
    group_return_spreads = [float(np.ptp(trajectory_return_delta[begin : begin + 4])) for begin in range(0, 32, 4)]

    gate = prereg["decision"]
    checks = {
        "exact_requests": len(rows) == 800,
        "exact_trajectories": len(trajectory_return_delta) == 32,
        "learned_right_routes": len(learned) >= gate["learned_right_routes_min"],
        "learned_request_mean_delta_positive": float(learned_delta.mean()) > 0.0,
        "learned_request_positive_fraction": float((learned_delta > 0).mean()) >= gate["learned_request_positive_fraction_min"],
        "trajectory_mean_return_delta_positive": float(trajectory_return_delta.mean()) > 0.0,
        "trajectory_positive_return_fraction": float((trajectory_return_delta > 0).mean()) >= gate["trajectory_positive_return_fraction_min"],
        "trajectory_groups_with_spread": sum(value >= gate["trajectory_group_return_spread_min"] for value in group_return_spreads) >= gate["trajectory_groups_with_spread_min"],
        "candidate_final_reward_mean_not_lower": float(candidate_by_time[-1].mean()) >= float(baseline_by_time[-1].mean()),
    }
    report = {
        "format": "strict-track2-v429-v426-full-trajectory-reward-trace32-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "requests": len(rows),
        "learned_right_routes": len(learned),
        "frozen_right_routes": sum(row["route"] == "frozen_right" for row in rows),
        "left_routes": sum(row["route"] == "left" for row in rows),
        "reward": {
            "v209_mean": float(baseline_score.mean()),
            "v426_mean": float(candidate_score.mean()),
            "all_mean_delta": float(delta.mean()),
            "learned_mean_delta": float(learned_delta.mean()),
            "learned_positive_delta_fraction": float((learned_delta > 0).mean()),
            "v209_hits_ge_0p9": int((baseline_score >= .9).sum()),
            "v426_hits_ge_0p9": int((candidate_score >= .9).sum()),
            "v209_final_mean": float(baseline_by_time[-1].mean()),
            "v426_final_mean": float(candidate_by_time[-1].mean()),
        },
        "trajectory": {
            "discount": gamma,
            "mean_return_delta": float(trajectory_return_delta.mean()),
            "median_return_delta": float(np.median(trajectory_return_delta)),
            "positive_return_fraction": float((trajectory_return_delta > 0).mean()),
            "return_delta": trajectory_return_delta.tolist(),
            "group_return_spread": group_return_spreads,
        },
        "checks": checks,
        "guards": {"policy_updates": 0, "checkpoint_writes": 0, "reward_is_official_training_signal_not_simulator_outcome": True, "hidden_or_final_data": False, "real_submission": False},
        "rows": [{**row, "v209_reward": float(baseline_score[index]), "v426_reward": float(candidate_score[index]), "reward_delta": float(delta[index])} for index, row in enumerate(rows)],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
