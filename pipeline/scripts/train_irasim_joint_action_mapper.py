#!/usr/bin/env python3
"""Fit the production-valid AgileX joint-command to Bridge-action mapper."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from wam_pipeline.irasim_action_mapper import JointActionMapper, joint_action_features
from wam_pipeline.irasim_physical_actions import _active_arm, _relative_ee_actions


def episode_examples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        joint = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        arm = _active_arm(joint)
        endpose = np.asarray(handle[f"endpose/{arm}_endpose"], dtype=np.float64)
        gripper = np.asarray(handle[f"joint_action/{arm}_gripper"], dtype=np.float64)
    inputs = joint_action_features(joint[:-1])
    outputs = _relative_ee_actions(endpose, gripper, -1.0 if arm == "left" else 1.0)[:, :7]
    return inputs, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(20260807)
    np.random.seed(20260807)
    split = json.loads(Path(args.split_manifest).read_text())
    root = Path(args.dataset)

    def load(episodes):
        values = [episode_examples(root / f"episode{episode}.hdf5") for episode in episodes]
        return np.concatenate([value[0] for value in values]), np.concatenate([value[1] for value in values])

    train_x, train_y = load(split["train_episodes"])
    validation_x, validation_y = load(split["validation_episodes"])
    device = torch.device(args.device)
    model = JointActionMapper().to(device)
    model.set_statistics(train_x, train_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    generator = torch.Generator(device=device).manual_seed(20260807)
    history = []
    best = None
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, len(x), (min(args.batch_size, len(x)),), generator=generator, device=device)
        prediction = model(x[indices])
        loss = model.normalized_loss(prediction, y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                val_prediction = model(torch.from_numpy(validation_x).to(device)).cpu().numpy()
            mae = np.abs(val_prediction - validation_y).mean(0)
            metrics = {"step": step, "train_loss": float(loss), "validation_mae_by_dimension": mae.tolist(), "validation_mae_mean": float(mae.mean())}
            history.append(metrics)
            if best is None or metrics["validation_mae_mean"] < best["metrics"]["validation_mae_mean"]:
                best = {"metrics": metrics, "state_dict": copy.deepcopy(model.state_dict())}
            print(json.dumps(metrics), flush=True)
            model.train()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best is None:
        raise RuntimeError("mapper training produced no validation checkpoint")
    torch.save(
        {
            "format": "track2-joint-to-bridge-action-mapper-v1",
            "state_dict": best["state_dict"],
            "best": best["metrics"],
            "history": history,
            "train_examples": len(train_x),
            "validation_examples": len(validation_x),
            "split_manifest": str(Path(args.split_manifest).resolve()),
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(best["metrics"], indent=2) + "\n")


if __name__ == "__main__":
    main()
