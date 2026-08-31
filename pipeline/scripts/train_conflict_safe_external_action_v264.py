#!/usr/bin/env python3
"""Learn external motion priors in the action embedding with conflict-safe gradients."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_autoregressive_unet import (
    WindowDataset, evaluate, frames_for_model, reconstruction_loss, rollout,
    save_checkpoint, window_motion_scores,
)
from wam_pipeline.autoregressive_unet import OneStepActionUNet
from wam_pipeline.external_robotwin_data_v260 import ExternalRandomizedWindowDataset, available_external_episodes


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def loss_for(model, batch, mean, std, device):
    context, history, future, target = batch
    context = frames_for_model(context).to(device, non_blocking=True)
    target = frames_for_model(target).to(device, non_blocking=True)
    history = ((history.to(device, non_blocking=True) - mean) / std).float()
    future = ((future.to(device, non_blocking=True) - mean) / std).float()
    prediction = rollout(model, context, history, future)
    previous = torch.cat([context[:, -1:], target[:, :-1]], dim=1)
    loss = reconstruction_loss(
        prediction, target, previous, motion_weight=2.5, motion_threshold=0.025,
        horizon_loss_power=0.35, temporal_delta_weight=0.35,
        texture_laplacian_weight=0.25,
    )
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--official-windows", required=True)
    parser.add_argument("--official-split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--official-batch-size", type=int, default=8)
    parser.add_argument("--external-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--external-gradient-weight", type=float, default=0.25)
    parser.add_argument("--pullback-weight", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--external-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=264)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(np.asarray(values["mean"], np.float32)).to(device)
        std = torch.from_numpy(np.asarray(values["std"], np.float32)).to(device)

    split = json.loads(Path(args.official_split).read_text())
    official = WindowDataset(Path(args.official_windows), split["train_episodes"])
    dev = WindowDataset(Path(args.official_windows), split["validation_episodes"])
    scores = window_motion_scores(official)
    weights = np.ones(len(official), np.float64); weights[scores >= 0.04] = 2.0
    official_sampler = WeightedRandomSampler(
        torch.from_numpy(weights), max(len(official), args.steps * args.official_batch_size),
        replacement=True, generator=torch.Generator().manual_seed(args.seed),
    )
    official_loader = DataLoader(
        official, batch_size=args.official_batch_size, sampler=official_sampler,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    external_ids = available_external_episodes(args.external_root)
    external = ExternalRandomizedWindowDataset(
        args.external_root, episodes=set(external_ids[:-50]), stride=args.external_stride,
    )
    external_loader = DataLoader(
        external, batch_size=args.external_batch_size, shuffle=True, num_workers=2,
        pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    dev_loader = DataLoader(dev, batch_size=10, shuffle=False, num_workers=2)

    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.requires_grad_(False); model.action_embedding.requires_grad_(True)
    parameters = list(model.action_embedding.parameters())
    anchors = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    baseline = evaluate(dev_loader, model, device, mean, std, 0.04)
    baseline_mae = float(baseline["rollout_mae"])
    baseline_high = float(baseline["high_motion_rollout_mae"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "checkpoint": str(checkpoint.resolve()), "official_train_windows": len(official),
        "external_train_windows": len(external), "trainable_parameters": sum(p.numel() for p in parameters),
        "algorithm": "PCGrad external gradient projected against official gradient; frozen visual backbone",
    })
    history = [{"step": 0, "validation": baseline}]
    metadata = {
        "backend": "autoregressive_unet", "format": "track2-autoregressive-unet-v1",
        "stage": "v26.4-conflict-safe-external-action", "checkpoint_step": 0,
        "config": config, "validation": history, "selection": "identity baseline",
    }
    save_checkpoint(output / "best", model, mean, std, metadata)
    print(json.dumps({"step": 0, "validation": baseline, "config": config}), flush=True)

    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    official_iterator, external_iterator = iter(official_loader), iter(external_loader)
    best = baseline_mae
    conflict_steps = 0
    for step in range(1, args.steps + 1):
        official_batch, official_iterator = next_batch(official_loader, official_iterator)
        external_batch, external_iterator = next_batch(external_loader, external_iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            official_loss = loss_for(model, official_batch, mean, std, device)
        official_grad = torch.autograd.grad(official_loss, parameters)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            external_loss = loss_for(model, external_batch, mean, std, device)
        external_grad = torch.autograd.grad(external_loss, parameters)
        dot = sum((left.float() * right.float()).sum() for left, right in zip(official_grad, external_grad))
        official_norm = sum((value.float() ** 2).sum() for value in official_grad).clamp_min(1e-12)
        projection = (dot / official_norm).clamp_max(0.0)
        if float(dot) < 0:
            conflict_steps += 1
        for parameter, anchor, left, right in zip(parameters, anchors, official_grad, external_grad):
            safe_external = right.float() - projection * left.float()
            parameter.grad = (
                left.float() + args.external_gradient_weight * safe_external
                + args.pullback_weight * (parameter.detach() - anchor)
            )
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({
                "step": step, "official_loss": float(official_loss), "external_loss": float(external_loss),
                "gradient_dot": float(dot), "conflict_fraction": conflict_steps / step,
            }), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(dev_loader, model, device, mean, std, 0.04); model.train()
            metric = float(result["rollout_mae"]); high = float(result["high_motion_rollout_mae"])
            safe = high <= baseline_high + 1e-9
            history.append({"step": step, "validation": result, "high_motion_safe": safe})
            if safe and metric < best:
                best = metric
                metadata = {
                    "backend": "autoregressive_unet", "format": "track2-autoregressive-unet-v1",
                    "stage": "v26.4-conflict-safe-external-action", "checkpoint_step": step,
                    "config": config, "validation": history, "selection": "official dev + high-motion non-regression",
                    "best_official_dev_mae": best,
                }
                save_checkpoint(output / "best", model, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "safe": safe, "best": best}), flush=True)
        if stop:
            break
    report = {
        "format": "track2-v26.4-conflict-safe-external-action-training", "config": config,
        "baseline": baseline, "best_official_dev_mae": best, "history": history,
        "conflict_step_fraction": conflict_steps / max(step, 1), "stopped_at_step": step,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "step": step, "best": best}), flush=True)


if __name__ == "__main__":
    main()
