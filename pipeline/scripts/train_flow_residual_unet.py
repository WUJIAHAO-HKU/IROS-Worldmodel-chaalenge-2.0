#!/usr/bin/env python3
"""Train an action-conditioned warp-and-residual Track 2 world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.flow_residual_unet import ActionConditionedFlowResidualUNet


class WindowDataset(Dataset):
    def __init__(self, directory: Path, episodes: list[int]) -> None:
        allowed = set(episodes)
        self.paths = [path for path in sorted(directory.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed]
        if not self.paths:
            raise ValueError("no windows selected")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["context_frames"].copy()),
                torch.cat([torch.from_numpy(data["history_actions"].copy()), torch.from_numpy(data["future_actions"].copy())]),
                torch.from_numpy(data["target_frames"].copy()),
            )


def frames_for_model(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 1, 4, 2, 3).float().div(255.0)


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    total = np.zeros(14, dtype=np.float64)
    squared = np.zeros(14, dtype=np.float64)
    count = 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            actions = np.concatenate([data["history_actions"], data["future_actions"]]).astype(np.float64)
        total += actions.sum(0)
        squared += np.square(actions).sum(0)
        count += len(actions)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = functional.l1_loss(prediction, target) + 0.1 * functional.mse_loss(prediction, target)
    pred_image, target_image = prediction.flatten(0, 1), target.flatten(0, 1)
    scale = functional.l1_loss(
        functional.interpolate(pred_image, scale_factor=0.5, mode="bilinear", align_corners=False),
        functional.interpolate(target_image, scale_factor=0.5, mode="bilinear", align_corners=False),
    )
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return pixel + 0.25 * scale + 0.2 * (edge_x + edge_y) / 2.0


def evaluate(loader, model, device, mean, std) -> dict[str, float]:
    model.eval()
    values = []
    with torch.no_grad():
        for context, actions, target in loader:
            context = frames_for_model(context).to(device)
            target = frames_for_model(target).to(device)
            prediction = model(context, ((actions.to(device) - mean) / std).float()).clamp(0, 1)
            values.append((
                float(functional.l1_loss(prediction, target).cpu()),
                float(functional.mse_loss(prediction, target).cpu()),
                float(functional.l1_loss(context[:, -1:].expand_as(target), target).cpu()),
            ))
    mae, mse, copy = np.mean(values, axis=0)
    return {"mae": float(mae), "mse": float(mse), "copy_last_mae": float(copy)}


def save_checkpoint(output: Path, model, mean, std, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-flow-residual-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_flow_residual_unet_config.npz", context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8), working_resolution=np.asarray(256), serving_resolution=np.asarray(256))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train, validation = WindowDataset(Path(args.windows), split["train_episodes"]), WindowDataset(Path(args.windows), split["validation_episodes"])
    device = torch.device(args.device)
    mean, std = action_statistics(train)
    mean, std = mean.to(device), std.to(device)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2, pin_memory=True)
    sample_count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, sample_count, dtype=np.int64).tolist()
    val_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = ActionConditionedFlowResidualUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output, best_mae, best_step, history = Path(args.output), float("inf"), 0, []
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            context, actions, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, actions, target = next(iterator)
        context, target = frames_for_model(context).to(device, non_blocking=True), frames_for_model(target).to(device, non_blocking=True)
        actions = ((actions.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context, actions)
        loss = reconstruction_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "mae": float(functional.l1_loss(prediction, target).detach().cpu())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(val_loader, model, device, mean, std)
            result["step"] = float(step)
            history.append(result)
            metadata = {"backend": "flow_residual_unet", "format": "track2-flow-residual-unet-v1", "windows": str(Path(args.windows).resolve()), "split_manifest": str(Path(args.split_manifest).resolve()), "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": sample_count, "context_frames": 5, "history_actions": 4, "future_actions": 8, "target_frames": 8, "action_dim": 14, "working_resolution": 256, "serving_resolution": 256, "checkpoint_step": step, "validation": history, "best_validation_mae": min(best_mae, result["mae"])}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, metadata)
            if result["mae"] < best_mae:
                best_mae, best_step = result["mae"], step
                metadata["best_checkpoint_step"] = best_step
                save_checkpoint(output, model, mean, std, metadata)
                save_checkpoint(output / "best", model, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
