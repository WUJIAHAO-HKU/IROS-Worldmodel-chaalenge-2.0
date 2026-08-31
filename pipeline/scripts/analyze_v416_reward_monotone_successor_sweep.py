#!/usr/bin/env python3
"""Discovery-only sweep of real public-demo successors on the v415 trace."""

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
        tensor = torch.from_numpy(frames[begin : begin + batch_size])
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255).to(device)
        with torch.inference_mode():
            reward = model.compute_reward(
                tensor, task_descriptions=prompts[begin : begin + batch_size]
            )
        values.append(reward.float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def load_trace(trace_dir):
    route_batches = [
        json.loads(line)["batch"]
        for line in (trace_dir / "route_trace.jsonl").read_text().splitlines()
    ]
    rows = [row for batch in route_batches for row in batch]
    frames, prompts = [], []
    for index, batch in enumerate(route_batches):
        stem = f"batch_{index:05d}"
        frame = np.load(trace_dir / f"{stem}_v400.npy", allow_pickle=False)
        text = json.loads((trace_dir / f"{stem}_prompts.json").read_text())
        if frame.shape[0] != len(batch) or len(text) != len(batch):
            raise RuntimeError(f"misaligned trace batch {index}")
        frames.append(frame)
        prompts.extend(text)
    frames = np.concatenate(frames)
    if len(rows) != 800 or frames.shape != (800, 256, 256, 3):
        raise RuntimeError("expected exact v415 trace")
    return rows, frames, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--v415-report", type=Path, required=True)
    parser.add_argument("--reward-map", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    v415 = json.loads(args.v415_report.read_text())
    if v415.get("format") != "strict-track2-v415-v409-reward-trace32-result-v1":
        raise RuntimeError("wrong v415 parent")
    rows, base_frames, prompts = load_trace(args.trace_dir)
    if len(v415["rows"]) != len(rows):
        raise RuntimeError("v415 row drift")
    base_score = np.asarray([row["v400_reward"] for row in v415["rows"]])
    expected_v409_score = np.asarray(
        [row["v409_reward"] for row in v415["rows"]], dtype=np.float64
    )

    with np.load(args.reward_map, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        episodes = values["episode_id"].astype(np.int64)
        starts = values["start"].astype(np.int64)
        clean = values["is_clean"].astype(bool)
        mapped_reward = values["reward"].astype(np.float32)
    mean_terminal_reward = np.nanmean(mapped_reward[:, :, -1], axis=1)
    episode_rows = {}
    for episode in np.unique(episodes[clean]):
        selected = np.flatnonzero(clean & (episodes == episode))
        episode_rows[int(episode)] = selected[np.argsort(starts[selected])]

    progressive = np.asarray(
        [index for index, row in enumerate(rows) if row["progressive_route"]],
        dtype=np.int64,
    )
    target_cache = {}

    def target_frame(row):
        row = int(row)
        if row not in target_cache:
            with np.load(paths[row], allow_pickle=False) as values:
                target_cache[row] = values["target_frames"][-1].copy()
        return target_cache[row]

    def choose(base, policy):
        episode = int(episodes[base])
        base_start = int(starts[base])
        candidates = episode_rows[episode]
        later = candidates[starts[candidates] >= base_start + 8]
        if not len(later):
            return None
        if policy.startswith("offset"):
            requested = base_start + int(policy.removeprefix("offset"))
            eligible = later[starts[later] >= requested]
            return int(eligible[0]) if len(eligible) else int(later[-1])
        if policy == "earliest_start96_within32":
            eligible = later[(starts[later] >= 96) & (starts[later] <= base_start + 32)]
            return int(eligible[0]) if len(eligible) else None
        if policy == "earliest_reward0p9_within32":
            eligible = later[
                (starts[later] <= base_start + 32)
                & (mean_terminal_reward[later] >= 0.9)
            ]
            return int(eligible[0]) if len(eligible) else None
        if policy == "max_reward_within32_ge0p9":
            eligible = later[starts[later] <= base_start + 32]
            if not len(eligible):
                return None
            best = int(eligible[np.argmax(mean_terminal_reward[eligible])])
            return best if mean_terminal_reward[best] >= 0.9 else None
        raise ValueError(policy)

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    policies = [
        "offset8",
        "offset16",
        "offset24",
        "offset32",
        "earliest_start96_within32",
        "earliest_reward0p9_within32",
        "max_reward_within32_ge0p9",
    ]
    multipliers = [0.5, 1.0, 1.5, 2.0]
    candidates = []
    for policy in policies:
        selected = {index: choose(rows[index]["progressive_base_row"], policy) for index in progressive}
        routed = np.asarray([index for index in progressive if selected[index] is not None])
        for multiplier in multipliers:
            images = []
            active_prompts = []
            alpha = []
            advances = []
            for index in routed:
                row = rows[index]
                target = selected[int(index)]
                blend = min(1.0, float(row["contracted_alpha"]) * multiplier)
                image = np.clip(
                    np.rint(
                        (1.0 - blend) * base_frames[index].astype(np.float32)
                        + blend * target_frame(target).astype(np.float32)
                    ),
                    0,
                    255,
                ).astype(np.uint8)
                images.append(image)
                active_prompts.append(prompts[index])
                alpha.append(blend)
                advances.append(int(starts[target] - starts[row["progressive_base_row"]]))
            candidate_score = base_score.copy()
            if len(routed):
                candidate_score[routed] = score(
                    reward_model,
                    np.stack(images),
                    active_prompts,
                    torch.device(args.device),
                    args.batch_size,
                )
            delta = candidate_score - base_score
            active_delta = delta[routed]
            group_spread = []
            group_reward_spread = []
            groups_with_route = []
            for begin in range(0, 800, 4):
                group = np.arange(begin, begin + 4)
                if np.any(np.isin(group, routed)):
                    groups_with_route.append(begin // 4)
                    group_spread.append(float(np.ptp(delta[group])))
                    group_reward_spread.append(float(np.ptp(candidate_score[group])))
            protected = base_score >= 0.9
            candidates.append(
                {
                    "policy": policy,
                    "alpha_multiplier": multiplier,
                    "routes": int(len(routed)),
                    "groups_with_route": len(groups_with_route),
                    "target_advance_mean": float(np.mean(advances)) if advances else None,
                    "target_advance_max": int(max(advances)) if advances else None,
                    "all_mean_delta": float(delta.mean()),
                    "route_mean_delta": float(active_delta.mean()) if len(routed) else None,
                    "route_median_delta": float(np.median(active_delta)) if len(routed) else None,
                    "route_positive_delta_fraction": float((active_delta > 0).mean()) if len(routed) else None,
                    "route_delta_gt_0p01": int((active_delta >= 0.01).sum()),
                    "route_delta_lt_minus_0p01": int((active_delta <= -0.01).sum()),
                    "groups_delta_spread_ge_0p05": int((np.asarray(group_spread) >= 0.05).sum()),
                    "groups_reward_spread_ge_0p05": int((np.asarray(group_reward_spread) >= 0.05).sum()),
                    "v400_hits_ge_0p9": int((base_score >= 0.9).sum()),
                    "candidate_hits_ge_0p9": int((candidate_score >= 0.9).sum()),
                    "protected_high_reward_min_delta": float(delta[protected].min()) if protected.any() else None,
                    "alpha_delta_correlation": (
                        float(np.corrcoef(np.asarray(alpha), active_delta)[0, 1])
                        if len(alpha) > 1 and np.std(alpha) and np.std(active_delta)
                        else 0.0
                    ),
                    "v409_reproduction_reward_mae": (
                        float(np.abs(candidate_score - expected_v409_score).mean())
                        if policy == "offset8" and multiplier == 1.0
                        else None
                    ),
                }
            )
            print(json.dumps(candidates[-1]), flush=True)
    report = {
        "format": "strict-track2-v416-reward-monotone-successor-discovery-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_v415_passed": v415["passed"],
        "protocol": {
            "discovery_trace_only": True,
            "candidate_policies": policies,
            "alpha_multipliers": multipliers,
            "selection_not_authorized_without_recursive_and_fresh_seed_confirmation": True,
        },
        "candidates": candidates,
        "guards": {
            "targets_are_real_official_public_demo_frames": True,
            "simulator_outcomes_used": False,
            "hidden_or_final_data": False,
            "policy_updates": 0,
            "checkpoint_writes": 0,
            "real_submission": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
