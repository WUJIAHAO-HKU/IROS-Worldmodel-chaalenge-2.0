#!/usr/bin/env python3
"""Train a Track 2 action-conditioned residual U-Net on episode-held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.residual_unet import ActionConditionedResidualUNet


class WindowDataset(Dataset):
    def __init__(self, directory: Path, episodes: list[int]) -> None:
        allowed = set(episodes)
        self.paths = [
            path for path in sorted(directory.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed
        ]
        if not self.paths:
            raise ValueError("no windows selected")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as data:
            context = torch.from_numpy(data["context_frames"].copy())
            history = torch.from_numpy(data["history_actions"].copy())
            future = torch.from_numpy(data["future_actions"].copy())
            target = torch.from_numpy(data["target_frames"].copy())
        return context, torch.cat([history, future]), target


def to_model_frames(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    batch, steps = frames.shape[:2]
    image = frames.permute(0, 1, 4, 2, 3).reshape(batch * steps, 3, 256, 256).float().div(255.0)
    return functional.interpolate(image, size=(resolution, resolution), mode="bilinear", align_corners=False).reshape(
        batch, steps, 3, resolution, resolution
    )


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    total = np.zeros(14, dtype=np.float64)
    squared = np.zeros(14, dtype=np.float64)
    count = 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            actions = np.concatenate([data["history_actions"], data["future_actions"]]).astype(np.float64)
        total += actions.sum(axis=0)
        squared += np.square(actions).sum(axis=0)
        count += len(actions)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def validation_indices(dataset: WindowDataset, count: int) -> list[int]:
    if count >= len(dataset):
        return list(range(len(dataset)))
    # File ordering groups episodes. Picking over the full span includes every
    # held-out episode in the fixed validation subset.
    return np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mae = functional.l1_loss(prediction, target)
    mse = functional.mse_loss(prediction, target)
    return mae, mse


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float,
    multi_scale_weight: float,
    edge_weight: float,
) -> torch.Tensor:
    """Preserve full-resolution pixels and object boundaries during fine-tuning."""
    mae, mse = metrics(prediction, target)
    loss = mae + mse_weight * mse
    if multi_scale_weight:
        coarse = 0.0
        pred_image = prediction.flatten(0, 1)
        target_image = target.flatten(0, 1)
        for scale in (0.5, 0.25):
            coarse += functional.l1_loss(
                functional.interpolate(pred_image, scale_factor=scale, mode="bilinear", align_corners=False),
                functional.interpolate(target_image, scale_factor=scale, mode="bilinear", align_corners=False),
            )
        loss = loss + multi_scale_weight * coarse / 2.0
    if edge_weight:
        horizontal = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
        vertical = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :])
        loss = loss + edge_weight * (horizontal + vertical) / 2.0
    return loss


def evaluate(loader, model, device, action_mean, action_std, resolution, max_batches) -> dict[str, float]:
    model.eval()
    values: list[tuple[float, float, float]] = []
    with torch.no_grad():
        for index, (context, actions, target) in enumerate(loader):
            if index >= max_batches:
                break
            context = to_model_frames(context, resolution).to(device)
            target = to_model_frames(target, resolution).to(device)
            actions = ((actions.to(device) - action_mean) / action_std).float()
            prediction = model(context, actions).clamp(0.0, 1.0)
            mae, mse = metrics(prediction, target)
            copy_mae = functional.l1_loss(context[:, -1:].expand_as(target), target)
            values.append((float(mae.cpu()), float(mse.cpu()), float(copy_mae.cpu())))
    if not values:
        raise RuntimeError("validation loader yielded no batches")
    result = np.asarray(values).mean(axis=0)
    return {"mae": float(result[0]), "mse": float(result[1]), "copy_last_mae": float(result[2])}


def save_checkpoint(
    output: Path,
    model,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    metadata: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-residual-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=action_mean.cpu().numpy(), std=action_std.cpu().numpy())
    np.savez(
        output / "track2_residual_unet_config.npz",
        context_frames=np.asarray(5),
        action_dim=np.asarray(14),
        prediction_frames=np.asarray(8),
        working_resolution=np.asarray(metadata["working_resolution"]),
        serving_resolution=np.asarray(256),
    )
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--resolution", type=int, default=128, choices=(64, 128, 256))
    parser.add_argument("--init-checkpoint", help="Existing compatible checkpoint directory for fine-tuning")
    parser.add_argument("--mse-loss-weight", type=float, default=0.25)
    parser.add_argument("--multi-scale-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validation_interval, args.validation_batches) < 1:
        raise SystemExit("steps, batch size and validation settings must be positive")

    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train = WindowDataset(Path(args.windows), split["train_episodes"])
    validation = WindowDataset(Path(args.windows), split["validation_episodes"])
    device = torch.device(args.device)
    action_mean, action_std = action_statistics(train)
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=2, pin_memory=True)
    sample_count = min(len(validation), args.validation_batches * args.batch_size)
    val_loader = DataLoader(
        Subset(validation, validation_indices(validation, sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = ActionConditionedResidualUNet().to(device)
    if args.init_checkpoint:
        state = torch.load(Path(args.init_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
        if state.get("format") != "track2-residual-unet-v1":
            raise SystemExit("--init-checkpoint is not a compatible Track 2 residual U-Net")
        model.load_state_dict(state["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    history: list[dict[str, float]] = []
    best_mae = float("inf")
    best_step = 0
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            context, actions, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, actions, target = next(iterator)
        context = to_model_frames(context, args.resolution).to(device, non_blocking=True)
        target = to_model_frames(target, args.resolution).to(device, non_blocking=True)
        actions = ((actions.to(device, non_blocking=True) - action_mean) / action_std).float()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context, actions)
        mae, mse = metrics(prediction, target)
        loss = reconstruction_loss(
            prediction,
            target,
            args.mse_loss_weight,
            args.multi_scale_loss_weight,
            args.edge_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "mae": float(mae.detach().cpu())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(val_loader, model, device, action_mean, action_std, args.resolution, args.validation_batches)
            result["step"] = float(step)
            history.append(result)
            metadata = {
                "backend": "residual_unet",
                "format": "track2-residual-unet-v1",
                "windows": str(Path(args.windows).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "train_window_count": len(train),
                "validation_window_count": len(validation),
                "validation_sample_count": sample_count,
                "context_frames": 5,
                "history_actions": 4,
                "future_actions": 8,
                "target_frames": 8,
                "action_dim": 14,
                "working_resolution": args.resolution,
                "serving_resolution": 256,
                "initialization_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
                "mse_loss_weight": args.mse_loss_weight,
                "multi_scale_loss_weight": args.multi_scale_loss_weight,
                "edge_loss_weight": args.edge_loss_weight,
                "checkpoint_step": step,
                "validation": history,
                "best_validation_mae": min(best_mae, result["mae"]),
            }
            checkpoint = output / "checkpoints" / f"checkpoint_step_{step:06d}"
            save_checkpoint(checkpoint, model, action_mean, action_std, metadata)
            if result["mae"] < best_mae:
                best_mae, best_step = result["mae"], step
                metadata["best_checkpoint_step"] = best_step
                save_checkpoint(output, model, action_mean, action_std, metadata)
                save_checkpoint(output / "best", model, action_mean, action_std, metadata)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
