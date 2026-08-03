#!/usr/bin/env python3
"""Train a detail refiner on predictions of a frozen 8-step Track 2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.autoregressive_unet_runtime import Track2AutoregressiveUNet
from wam_pipeline.refiner import Track2ImageRefiner


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


def loss_for(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = functional.l1_loss(prediction, target) + 0.05 * functional.mse_loss(prediction, target)
    prediction, target = prediction.flatten(0, 1), target.flatten(0, 1)
    coarse = functional.l1_loss(functional.avg_pool2d(prediction, 2), functional.avg_pool2d(target, 2))
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return pixel + 0.2 * coarse + 0.2 * (edge_x + edge_y) / 2.0


def frozen_predictions(runtime, context, history, future) -> torch.Tensor:
    values = [runtime.predict(c, h, f, 0, None) for c, h, f in zip(context.numpy(), history.numpy(), future.numpy())]
    return torch.from_numpy(np.stack(values)).permute(0, 1, 4, 2, 3).float().div(255.0)


def refine(refiner, prediction, last) -> torch.Tensor:
    batch, steps = prediction.shape[:2]
    last = last[:, None].expand(-1, steps, -1, -1, -1)
    return refiner(prediction.flatten(0, 1), last.flatten(0, 1)).reshape_as(prediction).clamp(0.0, 1.0)


def evaluate(loader, runtime, refiner, device) -> dict[str, float]:
    refiner.eval()
    values = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            prediction = frozen_predictions(runtime, context, history, future).to(device)
            target = to_image(target).to(device)
            refined = refine(refiner, prediction, to_image(context).to(device)[:, -1])
            values.append((float(functional.l1_loss(refined, target).float().cpu()), float(functional.l1_loss(prediction, target).float().cpu())))
    mae, base_mae = np.mean(values, axis=0)
    return {"mae": float(mae), "base_mae": float(base_mae)}


def save_checkpoint(output: Path, refiner, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-image-refiner-v1", "state_dict": refiner.state_dict()}, output / "refiner.pt")
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train = WindowDataset(Path(args.windows), split["train_episodes"])
    validation = WindowDataset(Path(args.windows), split["validation_episodes"])
    device = torch.device(args.device)
    runtime = Track2AutoregressiveUNet(args.base_checkpoint, device="cuda")
    runtime.model.eval()
    refiner = Track2ImageRefiner().to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    validation_loader = DataLoader(Subset(validation, np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    output, history, best_mae = Path(args.output), [], float("inf")
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            context, history_actions, future, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future, target = next(iterator)
        with torch.no_grad():
            prediction = frozen_predictions(runtime, context, history_actions, future).to(device, non_blocking=True)
        context = to_image(context).to(device, non_blocking=True)
        target = to_image(target).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            refined = refine(refiner, prediction, context[:, -1])
            loss = loss_for(refined, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "mae": float(functional.l1_loss(refined.float(), target).detach().cpu())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, runtime, refiner, device)
            refiner.train()
            result["step"] = float(step)
            history.append(result)
            metadata = {"format": "track2-image-refiner-v1", "base_checkpoint": str(Path(args.base_checkpoint).resolve()), "windows": str(Path(args.windows).resolve()), "split_manifest": str(Path(args.split_manifest).resolve()), "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": count, "checkpoint_step": step, "validation": history, "best_validation_mae": min(best_mae, result["mae"])}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", refiner, metadata)
            if result["mae"] < best_mae:
                best_mae = result["mae"]
                metadata["best_checkpoint_step"] = step
                save_checkpoint(output, refiner, metadata)
                save_checkpoint(output / "best", refiner, metadata)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
