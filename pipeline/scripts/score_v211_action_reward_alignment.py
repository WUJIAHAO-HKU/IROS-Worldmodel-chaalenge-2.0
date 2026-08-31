#!/usr/bin/env python3
"""Score captured v211 public rollouts and audit right-gripper reward sensitivity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def scalar_stats(values: np.ndarray) -> dict[str, float | int | None]:
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


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def to_nchw(frames: np.ndarray) -> torch.Tensor:
    frames = np.asarray(frames)
    if frames.ndim != 5:
        raise ValueError(f"predicted_frames must be rank 5, got {frames.shape}")
    if frames.shape[-1] == 3:
        tensor = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    elif frames.shape[2] == 3:
        tensor = torch.from_numpy(frames)
    else:
        raise ValueError(f"cannot locate RGB channel in {frames.shape}")
    tensor = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1]).float()
    if frames.dtype == np.uint8 or float(tensor.max()) > 1.2:
        tensor = tensor.div(255.0)
    elif float(tensor.min()) < -0.1:
        tensor = tensor.add(1.0).div(2.0)
    return tensor.clamp_(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--t5-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollout-epochs", type=int, default=2)
    args = ap.parse_args()

    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    if not files or len(files) % args.rollout_epochs:
        raise ValueError(
            f"capture count {len(files)} is not divisible by {args.rollout_epochs} epochs"
        )
    chunks_per_epoch = len(files) // args.rollout_epochs

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).eval().to(args.device)

    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_instructions: list[list[str]] = []
    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as item:
                actions = np.asarray(item["future_actions"], dtype=np.float32)
                frames = np.asarray(item["predicted_frames"])
                instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
            if actions.ndim != 3 or actions.shape[-1] != 14:
                raise ValueError(f"invalid action tensor in {path}: {actions.shape}")
            if len(instructions) != actions.shape[0]:
                raise ValueError(f"instruction/action batch mismatch in {path}")
            images = to_nchw(frames).to(args.device)
            expanded = [instruction for instruction in instructions for _ in range(actions.shape[1])]
            rewards = model.compute_reward(images, expanded).float().cpu().numpy()
            rewards = rewards.reshape(actions.shape[:2])
            if not np.isfinite(rewards).all():
                raise ValueError(f"non-finite rewards in {path}")
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_instructions.append(instructions)

    actions = np.stack(all_actions)
    rewards = np.stack(all_rewards)
    batch = actions.shape[1]
    trajectory_records: list[dict[str, object]] = []
    for epoch in range(args.rollout_epochs):
        start = epoch * chunks_per_epoch
        stop = start + chunks_per_epoch
        for env_index in range(batch):
            instruction_values = {
                row[env_index] for row in all_instructions[start:stop]
            }
            if len(instruction_values) != 1:
                raise ValueError(
                    f"instruction changed within epoch {epoch}, env {env_index}: "
                    f"{sorted(instruction_values)}"
                )
            instruction = next(iter(instruction_values))
            right_values = actions[start:stop, env_index, :, 13].reshape(-1)
            left_values = actions[start:stop, env_index, :, 6].reshape(-1)
            reward_values = rewards[start:stop, env_index, :].reshape(-1)
            trajectory_records.append(
                {
                    "epoch": epoch,
                    "env_index": env_index,
                    "instruction": instruction,
                    "right_close_fraction": float((right_values < 0.5).mean()),
                    "left_close_fraction": float((left_values < 0.5).mean()),
                    "terminal_reward": float(reward_values[-1]),
                    "peak_reward": float(reward_values.max()),
                    "mean_reward": float(reward_values.mean()),
                }
            )

    per_instruction: dict[str, object] = {}
    records_by_instruction: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in trajectory_records:
        records_by_instruction[str(record["instruction"])].append(record)
    flat_instructions = np.asarray(all_instructions, dtype=object)
    for instruction, records in sorted(records_by_instruction.items()):
        mask = flat_instructions == instruction
        action_values = actions[mask]
        reward_values = rewards[mask]
        close_mask = action_values[..., 13] < 0.5
        open_mask = ~close_mask
        close_rewards = reward_values[close_mask]
        open_rewards = reward_values[open_mask]
        trajectory_close = np.asarray([r["right_close_fraction"] for r in records])
        terminal_rewards = np.asarray([r["terminal_reward"] for r in records])
        per_instruction[instruction] = {
            "trajectories": len(records),
            "frame_reward": scalar_stats(reward_values),
            "reward_when_right_closed": scalar_stats(close_rewards),
            "reward_when_right_open": scalar_stats(open_rewards),
            "closed_minus_open_reward_mean": (
                float(close_rewards.mean() - open_rewards.mean())
                if close_rewards.size and open_rewards.size
                else None
            ),
            "trajectory_right_close_fraction": scalar_stats(trajectory_close),
            "terminal_reward": scalar_stats(terminal_rewards),
            "close_fraction_terminal_reward_correlation": correlation(
                trajectory_close, terminal_rewards
            ),
        }

    right_records = [
        record
        for record in trajectory_records
        if "right arm" in str(record["instruction"]).lower()
    ]
    right_close = np.asarray([r["right_close_fraction"] for r in right_records])
    right_terminal = np.asarray([r["terminal_reward"] for r in right_records])
    report = {
        "format": "strict-track2-v211-action-reward-alignment-audit-v1",
        "selection_use": False,
        "source": "public v209 world-model training rollouts captured before the sole update",
        "files": len(files),
        "action_shape": list(actions.shape),
        "reward_shape": list(rewards.shape),
        "reward": scalar_stats(rewards),
        "trajectory_records": trajectory_records,
        "per_instruction": per_instruction,
        "explicit_right_instruction": {
            "trajectories": len(right_records),
            "right_close_fraction": scalar_stats(right_close),
            "terminal_reward": scalar_stats(right_terminal),
            "close_fraction_terminal_reward_correlation": correlation(
                right_close, right_terminal
            ),
        },
        "guards": {
            "public_rollouts_only": True,
            "policy_modified": False,
            "world_model_modified": False,
            "candidate_selection": False,
            "final_128_evaluation": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
