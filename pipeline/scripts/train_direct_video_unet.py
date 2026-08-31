#!/usr/bin/env python3
"""Train a direct, horizon-conditioned Track 2 future-frame predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from wam_pipeline.direct_video_unet import DirectActionVideoUNet


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
                torch.from_numpy(data["history_actions"].copy()),
                torch.from_numpy(data["future_actions"].copy()),
                torch.from_numpy(data["target_frames"].copy()),
            )


def frames_for_model(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 1, 4, 2, 3).float().div(255.0)


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    total, squared, count = np.zeros(14, np.float64), np.zeros(14, np.float64), 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            actions = np.concatenate((data["history_actions"], data["future_actions"])).astype(np.float64)
        total += actions.sum(0)
        squared += np.square(actions).sum(0)
        count += len(actions)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def motion_scores(dataset: WindowDataset) -> np.ndarray:
    values = []
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            target = data["target_frames"].astype(np.float32) / 255.0
            previous = np.concatenate((data["context_frames"][-1:].astype(np.float32) / 255.0, target[:-1]), axis=0)
        values.append(float(np.abs(target - previous).mean()))
    return np.asarray(values, dtype=np.float64)


def weighted_sampler(dataset: WindowDataset, threshold: float, factor: float, seed: int) -> tuple[WeightedRandomSampler, dict[str, float | int]]:
    scores = motion_scores(dataset)
    high = scores >= threshold
    weights = np.ones(len(scores), dtype=np.float64)
    weights[high] = factor
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(dataset), replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    return sampler, {
        "threshold": float(threshold),
        "oversample_factor": float(factor),
        "high_motion_window_count": int(high.sum()),
        "high_motion_fraction": float(high.mean()),
        "expected_high_motion_draw_fraction": float(weights[high].sum() / weights.sum()) if high.any() else 0.0,
    }


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, last: torch.Tensor, motion_weight: float, motion_threshold: float) -> torch.Tensor:
    previous = torch.cat((last[:, None], target[:, :-1]), dim=1)
    changed = (target - previous).abs().mean(dim=2, keepdim=True) >= motion_threshold
    weights = 1.0 + (motion_weight - 1.0) * changed.to(target.dtype)
    pixel = (weights * (prediction - target).abs()).mean() + 0.03 * (weights * (prediction - target).square()).mean()
    prediction_flat, target_flat = prediction.flatten(0, 1), target.flatten(0, 1)
    coarse = functional.l1_loss(functional.avg_pool2d(prediction_flat, 2), functional.avg_pool2d(target_flat, 2))
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    # Motion-delta consistency prevents a low-motion average from hiding moving objects.
    delta = functional.l1_loss(prediction - last[:, None], target - last[:, None])
    return pixel + 0.15 * coarse + 0.12 * (edge_x + edge_y) / 2.0 + 0.20 * delta


def evaluate(loader, model, device, mean, std, accept_mae: float) -> dict:
    model.eval()
    errors = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            context = frames_for_model(context).to(device, non_blocking=True)
            target = frames_for_model(target).to(device, non_blocking=True)
            actions = torch.cat((history, future), dim=1).to(device, non_blocking=True)
            prediction = model(context, ((actions - mean) / std).float()).clamp(0.0, 1.0)
            errors.append((prediction.float() - target.float()).abs().mean(dim=(2, 3, 4)).cpu())
    error = torch.cat(errors)
    window_mean, window_peak = error.mean(dim=1), error.max(dim=1).values
    return {
        "mae": float(error.mean()),
        "mae_by_prediction_frame": [float(value) for value in error.mean(dim=0)],
        "max_window_mean_mae": float(window_mean.max()),
        "max_window_frame_mae": float(window_peak.max()),
        "windows_passing_mean": int((window_mean * 255.0 < accept_mae).sum()),
        "windows_passing_all_frames": int((window_peak * 255.0 < accept_mae).sum()),
        "window_count": int(len(error)),
        "accept_mae_0_255": float(accept_mae),
        "all_windows_and_frames_pass": bool((window_peak * 255.0 < accept_mae).all()),
    }


def save_checkpoint(output: Path, model, mean, std, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-direct-video-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_direct_video_unet_config.npz", context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8), working_resolution=np.asarray(256), serving_resolution=np.asarray(256))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--motion-weight", type=float, default=3.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=3.0)
    parser.add_argument("--accept-mae", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.accept_mae <= 0:
        raise SystemExit("steps, batch size, and --accept-mae must be positive")
    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train = WindowDataset(Path(args.windows), split["train_episodes"])
    validation = WindowDataset(Path(args.windows), split["validation_episodes"])
    device = torch.device(args.device)
    mean, std = action_statistics(train)
    mean, std = mean.to(device), std.to(device)
    sampler, sampling = weighted_sampler(train, args.high_motion_threshold, args.high_motion_oversample_factor, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = DirectActionVideoUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    output, history, best_mae = Path(args.output), [], float("inf")
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            context, history_actions, future, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        actions = torch.cat((history_actions, future), dim=1).to(device, non_blocking=True)
        actions = ((actions - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(context, actions).clamp(0.0, 1.0)
            loss = reconstruction_loss(prediction, target, context[:, -1], args.motion_weight, args.motion_threshold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "mae": float(functional.l1_loss(prediction.float(), target).detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.accept_mae)
            model.train()
            result["step"] = step
            history.append(result)
            metadata = {
                "backend": "direct-video-unet", "format": "track2-direct-video-unet-v1", "windows": str(Path(args.windows).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()), "train_window_count": len(train), "validation_window_count": len(validation),
                "validation_sample_count": count, "context_frames": 5, "history_actions": 4, "future_actions": 8, "target_frames": 8,
                "action_dim": 14, "working_resolution": 256, "serving_resolution": 256, "training_steps": args.steps, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold,
                "high_motion_sampling": sampling, "accept_mae_0_255": args.accept_mae, "checkpoint_step": step, "validation": history,
                "best_validation_mae": min(best_mae, result["mae"]),
            }
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, metadata)
            if result["mae"] < best_mae:
                best_mae = result["mae"]
                metadata["best_checkpoint_step"] = step
                save_checkpoint(output, model, mean, std, metadata)
                save_checkpoint(output / "best", model, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
