#!/usr/bin/env python3
"""Train the v17 action-to-image-pose forward-kinematics surrogate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import torch

try:
    from wam_pipeline.object_geometry_v170 import (
        ActionPoseProjector,
        denormalize_pose,
        normalize_action,
        project_endpose,
    )
except ModuleNotFoundError:  # Standalone mirror under oracle_stage/.
    from object_geometry_v170 import (
        ActionPoseProjector,
        denormalize_pose,
        normalize_action,
        project_endpose,
    )


def episode_number(path: Path) -> int:
    match = re.search(r"episode(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair action[t] with the calibrated pose visible in observation[t+1]."""
    actions, targets, arms = [], [], []
    with h5py.File(path, "r") as handle:
        joint = handle["joint_action/vector"][:-1].astype(np.float32)
        intrinsic = handle["observation/head_camera/intrinsic_cv"][1:]
        extrinsic = handle["observation/head_camera/extrinsic_cv"][1:]
        for arm, side in enumerate(("left", "right")):
            pose = handle[f"endpose/{side}_endpose"][1:]
            projected = np.stack(
                [project_endpose(pose[i], intrinsic[i], extrinsic[i]) for i in range(len(pose))]
            )
            actions.append(joint[:, arm * 7:(arm + 1) * 7])
            targets.append(projected)
            arms.append(np.full(len(projected), arm, dtype=np.int64))
    return np.concatenate(actions), np.concatenate(targets), np.concatenate(arms)


def load_split(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = [load_episode(path) for path in paths]
    return tuple(np.concatenate([value[index] for value in values]) for index in range(3))


def arm_stats(value: np.ndarray, arms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.stack([value[arms == arm].mean(0) for arm in (0, 1)]).astype(np.float32)
    std = np.stack([value[arms == arm].std(0) for arm in (0, 1)]).astype(np.float32)
    return mean, np.maximum(std, 1e-4)


@torch.inference_mode()
def evaluate(model: ActionPoseProjector, values: tuple[torch.Tensor, ...],
             stats: tuple[torch.Tensor, ...], batch_size: int) -> dict:
    action, target, arms = values
    action_mean, action_std, target_mean, target_std = stats
    predictions = []
    for start in range(0, len(action), batch_size):
        selected = slice(start, start + batch_size)
        normalized = normalize_action(action[selected], action_mean, action_std, arms[selected])
        prediction = model(normalized, arms[selected])
        predictions.append(denormalize_pose(prediction, target_mean, target_std, arms[selected]))
    prediction = torch.cat(predictions)
    error = (prediction - target).abs()
    result = {
        "sample_count": int(len(target)),
        "landmark_mae_px": float(error[:, :6].mean()),
        "origin_mae_px": float(error[:, :2].mean()),
        "axis_endpoint_mae_px": float(error[:, 2:6].mean()),
        "depth_mae_m": float(error[:, 6].mean()),
        "arms": {},
    }
    for arm in (0, 1):
        selected = arms == arm
        local = error[selected]
        result["arms"][f"arm{arm}"] = {
            "sample_count": int(selected.sum()),
            "landmark_mae_px": float(local[:, :6].mean()),
            "origin_mae_px": float(local[:, :2].mean()),
            "axis_endpoint_mae_px": float(local[:, 2:6].mean()),
            "depth_mae_m": float(local[:, 6].mean()),
        }
    return result


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-episodes", default="36,47")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    dev_episodes = {int(value) for value in args.dev_episodes.split(",") if value}
    train_episodes = set(split["train_episodes"]) - dev_episodes
    data = Path(args.dataset)
    train_paths = [data / f"episode{episode}.hdf5" for episode in sorted(train_episodes)]
    dev_paths = [data / f"episode{episode}.hdf5" for episode in sorted(dev_episodes)]
    train = load_split(train_paths); dev = load_split(dev_paths)
    action_mean, action_std = arm_stats(train[0], train[2])
    target_mean, target_std = arm_stats(train[1], train[2])

    device = torch.device(args.device)
    train_tensors = tuple(torch.from_numpy(value).to(device) for value in train)
    dev_tensors = tuple(torch.from_numpy(value).to(device) for value in dev)
    stats = tuple(torch.from_numpy(value).to(device) for value in
                  (action_mean, action_std, target_mean, target_std))
    model = ActionPoseProjector(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best = float("inf"); history = []

    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train_tensors[0]), (args.batch_size,), generator=generator,
                                device=device)
        action, target, arms = (value[indices] for value in train_tensors)
        normalized_action = normalize_action(action, stats[0], stats[1], arms)
        normalized_target = (target - stats[2][arms]) / stats[3][arms]
        prediction = model(normalized_action, arms)
        # Projected landmarks are primary; depth prevents scale ambiguity.
        loss_landmark = torch.nn.functional.smooth_l1_loss(prediction[:, :6], normalized_target[:, :6])
        loss_depth = torch.nn.functional.smooth_l1_loss(prediction[:, 6], normalized_target[:, 6])
        loss = loss_landmark + 0.20 * loss_depth
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            model.eval(); metrics = evaluate(model, dev_tensors, stats, args.batch_size); model.train()
            record = {"step": step, "loss": float(loss), "dev": metrics}
            history.append(record); print(json.dumps(record), flush=True)
            score = max(value["landmark_mae_px"] for value in metrics["arms"].values())
            payload = {
                "format": "track2-action-pose-projector-v17.0",
                "step": step,
                "model": model.state_dict(),
                "hidden": args.hidden,
                "action_mean": action_mean,
                "action_std": action_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "dev_metrics": metrics,
            }
            torch.save(payload, output / "latest.pt")
            if score < best:
                best = score; torch.save(payload, output / "best.pt")

    manifest = {
        "format": "track2-action-pose-projector-v17.0-training",
        "train_episodes": sorted(train_episodes),
        "dev_episodes": sorted(dev_episodes),
        "train_samples": int(len(train[0])),
        "dev_samples": int(len(dev[0])),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_worst_arm_landmark_mae_px": best,
        "history": history,
    }
    atomic_json(output / "training_manifest.json", manifest)


if __name__ == "__main__":
    main()
