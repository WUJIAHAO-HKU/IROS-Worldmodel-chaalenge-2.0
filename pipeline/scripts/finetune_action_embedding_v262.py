#!/usr/bin/env python3
"""Official-domain fine-tuning of only the externally improved action embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_autoregressive_unet import (
    WindowDataset,
    evaluate,
    frames_for_model,
    reconstruction_loss,
    rollout,
    save_checkpoint,
    window_motion_scores,
)
from wam_pipeline.autoregressive_unet import OneStepActionUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=262)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise SystemExit("checkpoint format mismatch")
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as data:
        mean = torch.from_numpy(np.asarray(data["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(data["std"], dtype=np.float32)).to(device)
    split = json.loads(Path(args.split).read_text())
    train = WindowDataset(Path(args.windows), split["train_episodes"])
    validation = WindowDataset(Path(args.windows), split["validation_episodes"])
    scores = window_motion_scores(train)
    weights = np.ones(len(train), dtype=np.float64)
    weights[scores >= 0.04] = 2.0
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=max(len(train), args.steps * args.batch_size), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_embedding.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.action_embedding.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    output = Path(args.output)
    history = []
    baseline = evaluate(validation_loader, model, device, mean, std, 0.04)
    best = float(baseline["rollout_mae"])
    history.append({"step": 0, "validation": baseline})
    metadata = {
        "backend": "autoregressive_unet",
        "format": "track2-autoregressive-unet-v1",
        "stage": "v26.2-action-embedding-only-official-finetune",
        "checkpoint_step": 0,
        "initialization": str(checkpoint.resolve()),
        "trainable_parameters": trainable,
        "frozen_visual_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "validation": history,
        "best_official_dev_mae": best,
    }
    save_checkpoint(output / "best", model, mean, std, metadata)
    print(json.dumps({"step": 0, "validation": baseline, "trainable_parameters": trainable}), flush=True)
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            context, action_history, future, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, action_history, future, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        action_history = ((action_history.to(device, non_blocking=True) - mean) / std).float()
        future = ((future.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = rollout(model, context, action_history, future)
            previous = torch.cat([context[:, -1:], target[:, :-1]], dim=1)
            loss = reconstruction_loss(
                prediction, target, previous,
                motion_weight=2.5, motion_threshold=0.025, horizon_loss_power=0.35,
                temporal_delta_weight=0.35, texture_laplacian_weight=0.25,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.action_embedding.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, 0.04)
            model.train()
            history.append({"step": step, "validation": result})
            metric = float(result["rollout_mae"])
            metadata = {
                "backend": "autoregressive_unet", "format": "track2-autoregressive-unet-v1",
                "stage": "v26.2-action-embedding-only-official-finetune", "checkpoint_step": step,
                "initialization": str(checkpoint.resolve()), "trainable_parameters": trainable,
                "frozen_visual_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                "validation": history, "best_official_dev_mae": min(best, metric),
            }
            save_checkpoint(output / "latest", model, mean, std, metadata)
            if metric < best:
                best = metric
                save_checkpoint(output / "best", model, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "best": best}), flush=True)


if __name__ == "__main__":
    main()
