#!/usr/bin/env python3
"""Replay captured public actions through a Track-2 service and audit reward ranking."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or first.std() == 0.0 or second.std() == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def to_nchw(frames: np.ndarray) -> torch.Tensor:
    frames = np.asarray(frames)
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"expected [B,T,H,W,3] frames, got {frames.shape}")
    return (
        torch.from_numpy(frames)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 3, frames.shape[2], frames.shape[3])
        .float()
        .div_(255.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise ValueError("no rollout captures found")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint),
        config={"t5_model_name": str(args.t5_model)},
    ).eval().to(args.device)
    client = Track2ServiceClient(args.url, args.token, args.model_version)
    client.assert_ready()

    action_rows = []
    old_reward_rows = []
    new_reward_rows = []
    instruction_rows = []
    with torch.inference_mode():
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as item:
                contexts = item["context_frames"].copy()
                histories = item["history_actions"].astype(np.float32).copy()
                actions = item["future_actions"].astype(np.float32).copy()
                old_frames = item["predicted_frames"].copy()
                seeds = item["seeds"].astype(np.int64).copy()
                instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
            new_frames = client.predict_batch(
                contexts, histories, actions, seeds, instructions
            )
            expanded = [text for text in instructions for _ in range(actions.shape[1])]
            old_rewards = model.compute_reward(
                to_nchw(old_frames).to(args.device), expanded
            ).float().cpu().numpy().reshape(actions.shape[:2])
            new_rewards = model.compute_reward(
                to_nchw(new_frames).to(args.device), expanded
            ).float().cpu().numpy().reshape(actions.shape[:2])
            action_rows.append(actions)
            old_reward_rows.append(old_rewards)
            new_reward_rows.append(new_rewards)
            instruction_rows.append(instructions)
            print(json.dumps({"completed": index + 1, "total": len(files)}), flush=True)

    actions = np.stack(action_rows)
    old_rewards = np.stack(old_reward_rows)
    new_rewards = np.stack(new_reward_rows)
    instructions = np.asarray(instruction_rows, dtype=object)
    terminal_old = old_rewards[..., -1]
    terminal_new = new_rewards[..., -1]
    right_close = (actions[..., 13] < 0.5).mean(axis=-1)
    right_motion = np.linalg.norm(np.diff(actions[..., 7:13], axis=2), axis=-1).mean(axis=-1)

    group_rows = []
    for chunk in range(actions.shape[0]):
        for begin in range(0, actions.shape[1], args.group_size):
            stop = begin + args.group_size
            if stop > actions.shape[1]:
                continue
            group_rows.append(
                {
                    "chunk": chunk,
                    "begin": begin,
                    "instruction": str(instructions[chunk, begin]),
                    "old_terminal_std": float(terminal_old[chunk, begin:stop].std()),
                    "new_terminal_std": float(terminal_new[chunk, begin:stop].std()),
                    "new_terminal_range": float(
                        np.ptp(terminal_new[chunk, begin:stop])
                    ),
                    "right_close_terminal_correlation": correlation(
                        right_close[chunk, begin:stop],
                        terminal_new[chunk, begin:stop],
                    ),
                    "right_motion_terminal_correlation": correlation(
                        right_motion[chunk, begin:stop],
                        terminal_new[chunk, begin:stop],
                    ),
                }
            )

    per_instruction = {}
    for instruction in sorted(set(instructions.reshape(-1))):
        mask = instructions == instruction
        per_instruction[str(instruction)] = {
            "chunks": int(mask.sum()),
            "old_terminal_reward": stats(terminal_old[mask]),
            "new_terminal_reward": stats(terminal_new[mask]),
            "new_right_close_terminal_correlation": correlation(
                right_close[mask], terminal_new[mask]
            ),
            "new_right_motion_terminal_correlation": correlation(
                right_motion[mask], terminal_new[mask]
            ),
        }

    group_new_std = np.asarray([row["new_terminal_std"] for row in group_rows])
    group_old_std = np.asarray([row["old_terminal_std"] for row in group_rows])
    report = {
        "format": "strict-track2-service-action-reward-ranking-audit-v1",
        "source": "public v211 captured world-model rollouts",
        "service_model_version": args.model_version,
        "capture_files": len(files),
        "action_shape": list(actions.shape),
        "reward_shape": list(new_rewards.shape),
        "old_frame_reward": stats(old_rewards),
        "new_frame_reward": stats(new_rewards),
        "old_group_terminal_std": stats(group_old_std),
        "new_group_terminal_std": stats(group_new_std),
        "new_zero_std_group_fraction": float((group_new_std == 0.0).mean()),
        "new_right_close_terminal_correlation": correlation(right_close, terminal_new),
        "new_right_motion_terminal_correlation": correlation(right_motion, terminal_new),
        "per_instruction": per_instruction,
        "group_rows": group_rows,
        "guards": {
            "public_capture_only": True,
            "policy_modified": False,
            "candidate_selection": False,
            "final_128_used": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("group_rows", "per_instruction")}, indent=2))


if __name__ == "__main__":
    main()
