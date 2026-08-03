#!/usr/bin/env python3
"""Sharpen a Track 2 one-step predictor with conditional PatchGAN supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.autoregressive_unet import OneStepActionUNet
from wam_pipeline.refiner import ConditionalPatchDiscriminator


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
            return data["context_frames"].copy(), data["history_actions"].copy(), data["future_actions"].copy(), data["target_frames"].copy()


def to_image(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).float().div(255.0)


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    total, squared, count = np.zeros(14, np.float64), np.zeros(14, np.float64), 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            actions = np.concatenate([data["history_actions"], data["future_actions"]]).astype(np.float64)
        total += actions.sum(0)
        squared += np.square(actions).sum(0)
        count += len(actions)
    mean = total / count
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(np.maximum(squared / count - mean**2, 1e-8)).astype(np.float32))


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = functional.l1_loss(prediction, target) + 0.05 * functional.mse_loss(prediction, target)
    coarse = functional.l1_loss(functional.avg_pool2d(prediction, 2), functional.avg_pool2d(target, 2))
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return pixel + 0.2 * coarse + 0.2 * (edge_x + edge_y) / 2.0


def rollout(model, context, history, future) -> torch.Tensor:
    predictions = []
    for action in future.unbind(1):
        next_frame = model(context, torch.cat([history, action[:, None]], dim=1)).clamp(0.0, 1.0)
        predictions.append(next_frame)
        context, history = torch.cat([context[:, 1:], next_frame[:, None]], 1), torch.cat([history[:, 1:], action[:, None]], 1)
    return torch.stack(predictions, 1)


def evaluate(loader, model, device, mean, std) -> dict[str, float]:
    model.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            context = to_image(context).to(device)
            target = to_image(target).to(device)
            history = ((history.to(device) - mean) / std).float()
            future = ((future.to(device) - mean) / std).float()
            prediction = rollout(model, context, history, future)
            values.append(float(functional.l1_loss(prediction, target).float().cpu()))
    return {"rollout_mae": float(np.mean(values))}


def save_checkpoint(output: Path, model, discriminator, mean, std, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-autoregressive-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    torch.save({"format": "track2-conditional-patchgan-v1", "state_dict": discriminator.state_dict()}, output / "discriminator.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_autoregressive_unet_config.npz", context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8), working_resolution=np.asarray(256), serving_resolution=np.asarray(256))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adversarial-weight", type=float, default=0.01)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train, validation = WindowDataset(Path(args.windows), split["train_episodes"]), WindowDataset(Path(args.windows), split["validation_episodes"])
    device = torch.device(args.device)
    mean, std = action_statistics(train)
    mean, std = mean.to(device), std.to(device)
    state = torch.load(Path(args.init_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise SystemExit("--init-checkpoint must be an autoregressive Track 2 model")
    model, discriminator = OneStepActionUNet().to(device), ConditionalPatchDiscriminator().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.learning_rate * 2, betas=(0.5, 0.999), weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    validation_loader = DataLoader(Subset(validation, np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    output, history, best_mae = Path(args.output), [], float("inf")
    iterator = iter(loader)
    bce = nn.BCEWithLogitsLoss()
    for step in range(1, args.steps + 1):
        try:
            context, history_actions, future, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            context, history_actions, future, target = next(iterator)
        context = to_image(context).to(device, non_blocking=True)
        target = to_image(target).to(device, non_blocking=True)
        history_actions = ((history_actions.to(device, non_blocking=True) - mean) / std).float()
        future = ((future.to(device, non_blocking=True) - mean) / std).float()
        previous, real = context[:, -1], target[:, 0]
        optimizer_d.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fake = model(context, torch.cat([history_actions, future[:, :1]], 1)).clamp(0.0, 1.0)
            real_logits, fake_logits = discriminator(previous, real), discriminator(previous, fake.detach())
            d_loss = (bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))) / 2
        d_loss.backward()
        optimizer_d.step()
        optimizer_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fake = model(context, torch.cat([history_actions, future[:, :1]], 1)).clamp(0.0, 1.0)
            g_adv = bce(discriminator(previous, fake), torch.ones_like(real_logits))
            g_loss = reconstruction_loss(fake, real) + args.adversarial_weight * g_adv
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_g.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "g_loss": float(g_loss.detach().cpu()), "d_loss": float(d_loss.detach().cpu()), "mae": float(functional.l1_loss(fake.float(), real).detach().cpu())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std)
            model.train(); discriminator.train()
            result["step"] = float(step)
            history.append(result)
            metadata = {"format": "track2-autoregressive-unet-v1", "base_checkpoint": str(Path(args.init_checkpoint).resolve()), "adversarial_weight": args.adversarial_weight, "windows": str(Path(args.windows).resolve()), "split_manifest": str(Path(args.split_manifest).resolve()), "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": count, "checkpoint_step": step, "validation": history, "best_rollout_mae": min(best_mae, result["rollout_mae"])}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, discriminator, mean, std, metadata)
            if result["rollout_mae"] < best_mae:
                best_mae = result["rollout_mae"]
                metadata["best_checkpoint_step"] = step
                save_checkpoint(output, model, discriminator, mean, std, metadata)
                save_checkpoint(output / "best", model, discriminator, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
