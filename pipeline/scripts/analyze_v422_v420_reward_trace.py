#!/usr/bin/env python3
"""Score captured v400/v420 terminal frames with the official training reward."""

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
        batch = batch.float().div(255.0).to(device)
        with torch.inference_mode():
            result = model.compute_reward(batch, prompts[begin : begin + batch_size])
        values.append(result.float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def correlation(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True, type=Path)
    p.add_argument("--preregistration", required=True, type=Path)
    p.add_argument("--reward-checkpoint", required=True, type=Path)
    p.add_argument("--t5-model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v422-v420-reward-trace32-preregistration-v1":
        raise RuntimeError("wrong v422 preregistration")
    if args.output.exists():
        raise FileExistsError(args.output)
    route_batches = [json.loads(line)["batch"] for line in (args.trace_dir / "route_trace.jsonl").read_text().splitlines()]
    v400, v420, prompts = [], [], []
    for index, routes in enumerate(route_batches):
        stem = f"batch_{index:05d}"
        left = np.load(args.trace_dir / f"{stem}_v400.npy", allow_pickle=False)
        right = np.load(args.trace_dir / f"{stem}_v420.npy", allow_pickle=False)
        text = json.loads((args.trace_dir / f"{stem}_prompts.json").read_text())
        if left.shape != right.shape or left.shape[0] != len(routes) or len(text) != len(routes):
            raise RuntimeError(f"misaligned trace batch {index}")
        v400.append(left); v420.append(right); prompts.extend(text)
    v400, v420 = np.concatenate(v400), np.concatenate(v420)
    rows = [row for batch in route_batches for row in batch]
    if len(rows) != 800 or v400.shape != (800, 256, 256, 3) or v420.shape != v400.shape:
        raise RuntimeError("expected exact 800-request trace")
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}).to(args.device).eval().requires_grad_(False)
    base_score = score(reward, v400, prompts, torch.device(args.device), args.batch_size)
    candidate_score = score(reward, v420, prompts, torch.device(args.device), args.batch_size)
    delta = candidate_score - base_score
    progressive = np.asarray([i for i, row in enumerate(rows) if row["progressive_route"]], dtype=np.int64)
    alpha = np.asarray([row["contracted_alpha"] for row in rows], dtype=np.float64)
    groups, offset = [], 0
    for batch in route_batches:
        if len(batch) != 8: raise RuntimeError("non-native trace batch")
        for begin in (0, 4):
            indices = np.arange(offset + begin, offset + begin + 4)
            groups.append({"progressive": bool(any(rows[i]["progressive_route"] for i in indices)), "delta_spread": float(np.ptp(delta[indices])), "reward_spread": float(np.ptp(candidate_score[indices]))})
        offset += 8
    active_groups = [g for g in groups if g["progressive"]]
    delta_groups = [g for g in active_groups if g["delta_spread"] >= 0.05]
    d, a = delta[progressive], alpha[progressive]
    gate = prereg["decision"]
    checks = {
        "exact_requests": len(rows) == 800,
        "exact_groups": len(groups) == 200,
        "progressive_routes": len(progressive) >= gate["progressive_routes_min"],
        "progressive_mean_reward_delta_positive": float(d.mean()) > 0.0,
        "progressive_positive_delta_fraction": float((d > 0).mean()) >= gate["positive_reward_delta_fraction_min"],
        "alpha_reward_delta_correlation": correlation(a, d) >= gate["alpha_reward_delta_correlation_min"],
        "groups_with_delta_spread": len(delta_groups) >= gate["groups_with_delta_spread_ge_0p05_min"],
    }
    report = {
        "format": "strict-track2-v422-v420-reward-trace32-result-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "requests": len(rows),
        "progressive_routes": len(progressive),
        "terminal_precedence_routes": sum(row["terminal_precedence"] for row in rows),
        "reward": {"v400_mean": float(base_score.mean()), "v420_mean": float(candidate_score.mean()), "all_mean_delta": float(delta.mean()), "progressive_mean_delta": float(d.mean()), "progressive_median_delta": float(np.median(d)), "progressive_positive_delta_fraction": float((d > 0).mean()), "progressive_alpha_delta_correlation": correlation(a, d), "v400_hits_ge_0p9": int((base_score >= .9).sum()), "v420_hits_ge_0p9": int((candidate_score >= .9).sum())},
        "groups_total": len(groups), "groups_with_progressive": len(active_groups), "groups_with_progressive_and_delta_spread_ge_0p05": len(delta_groups),
        "checks": checks,
        "guards": {"policy_updates": 0, "checkpoint_writes": 0, "reward_is_official_training_signal_not_simulator_outcome": True, "hidden_or_final_data": False, "real_submission": False},
        "rows": [{**row, "v400_reward": float(base_score[i]), "v420_reward": float(candidate_score[i]), "reward_delta": float(delta[i])} for i, row in enumerate(rows)],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
