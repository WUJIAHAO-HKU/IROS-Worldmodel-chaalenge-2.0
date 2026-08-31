#!/usr/bin/env python3
"""Fine-tune the AR parent with an explicit dark-structure generation task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir))

from train_autoregressive_unet import (
    WindowDataset,
    cached_training_statistics,
    frames_for_model,
    high_motion_sampling,
    reconstruction_loss,
)
from wam_pipeline.autoregressive_structure_unet import OneStepActionSeparatedStructureUNet, load_separated_rgb_parent


def rollout(model, context, history, future):
    predictions, logits = [], []
    for action in future.unbind(dim=1):
        prediction, structure = model(context, torch.cat((history, action[:, None]), dim=1), return_structure=True)
        prediction = prediction.clamp(0.0, 1.0)
        predictions.append(prediction)
        logits.append(structure)
        # Keep the RGB parent's recurrent state pristine. The explicit repair
        # is emitted to the caller but is not recursively amplified.
        recurrent = (prediction + model.structure_correction(structure)).clamp(0.0, 1.0)
        context = torch.cat((context[:, 1:], recurrent[:, None]), dim=1)
        history = torch.cat((history[:, 1:], action[:, None]), dim=1)
    return torch.stack(predictions, dim=1), torch.stack(logits, dim=1)


def highpass(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape[:2]
    blurred = functional.avg_pool2d(value.flatten(0, 1), 5, stride=1, padding=2, count_include_pad=False).unflatten(0, shape)
    return value - blurred


def structure_target(target: torch.Tensor, prediction: torch.Tensor | None = None) -> torch.Tensor:
    target_luminance = target.mean(dim=2, keepdim=True)
    target_dark = torch.sigmoid((0.38 - target_luminance) * 24.0)
    if prediction is None:
        return target_dark
    # Supervise only dark geometry the RGB parent currently makes too bright.
    # This excludes already-correct black regions and background shadows.
    missing_dark = torch.sigmoid((prediction.detach().mean(dim=2, keepdim=True) - target_luminance - 0.01) * 32.0)
    return target_dark * missing_dark


def structure_loss(prediction, logits, target, mask_weight: float, mask_edge_weight: float, dark_rgb_weight: float):
    target_mask = structure_target(target, prediction)
    mask = torch.sigmoid(logits)
    mask_bce = functional.binary_cross_entropy_with_logits(logits, target_mask)
    mask_edge = 0.5 * (
        functional.l1_loss(mask[..., 1:] - mask[..., :-1], target_mask[..., 1:] - target_mask[..., :-1])
        + functional.l1_loss(mask[..., 1:, :] - mask[..., :-1, :], target_mask[..., 1:, :] - target_mask[..., :-1, :])
    )
    dark_rgb = (target_mask * (prediction - target).abs()).sum() / (target_mask.sum() * 3.0).clamp_min(1.0)
    total = mask_weight * mask_bce + mask_edge_weight * mask_edge + dark_rgb_weight * dark_rgb
    return total, {"mask_bce": mask_bce, "mask_edge": mask_edge, "dark_rgb": dark_rgb, "mask_mean": mask.mean(), "target_mask_mean": target_mask.mean()}


@torch.inference_mode()
def evaluate(loader, model, device, mean, std, high_motion_threshold: float, use_autocast: bool = True) -> dict[str, float]:
    sums = {name: 0.0 for name in ("rgb", "high_rgb", "temporal", "highpass", "edge", "dark", "mask_bce")}
    counts = {name: 0.0 for name in ("rgb", "high_rgb", "temporal", "highpass", "edge", "dark", "mask")}
    model.eval()
    for context, history, future, target in loader:
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        history = ((history.to(device, non_blocking=True) - mean) / std).float()
        future = ((future.to(device, non_blocking=True) - mean) / std).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast and device.type == "cuda"):
            prediction, logits = rollout(model, context, history, future)
        prediction, logits = prediction.float(), logits.float()
        previous_target = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
        previous_prediction = torch.cat((context[:, -1:], prediction[:, :-1]), dim=1)
        motion = (target - previous_target).abs().mean(dim=(1, 2, 3, 4))
        high = motion >= high_motion_threshold
        error = (prediction - target).abs()
        temporal = ((prediction - previous_prediction) - (target - previous_target)).abs()
        high_error = (highpass(prediction) - highpass(target)).abs()
        edge_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
        edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
        dark = target.mean(dim=2, keepdim=True) < 0.30
        dark_channels = dark.expand_as(target)
        mask_target = structure_target(target, prediction)
        sums["rgb"] += float(error.sum())
        sums["high_rgb"] += float(error[high].sum())
        sums["temporal"] += float(temporal.sum())
        sums["highpass"] += float(high_error.sum())
        sums["edge"] += float(edge_x.sum() + edge_y.sum())
        sums["dark"] += float(error[dark_channels].sum())
        sums["mask_bce"] += float(functional.binary_cross_entropy_with_logits(logits, mask_target, reduction="sum"))
        counts["rgb"] += error.numel()
        counts["high_rgb"] += int(high.sum()) * int(np.prod(target.shape[1:]))
        counts["temporal"] += temporal.numel()
        counts["highpass"] += high_error.numel()
        counts["edge"] += edge_x.numel() + edge_y.numel()
        counts["dark"] += int(dark_channels.sum())
        counts["mask"] += mask_target.numel()
    return {
        "rgb_mae": sums["rgb"] / counts["rgb"],
        "high_motion_rgb_mae": sums["high_rgb"] / max(counts["high_rgb"], 1),
        "temporal_delta_mae": sums["temporal"] / counts["temporal"],
        "highpass_mae": sums["highpass"] / counts["highpass"],
        "edge_mae": sums["edge"] / counts["edge"],
        "dark_region_rgb_mae": sums["dark"] / max(counts["dark"], 1),
        "structure_mask_bce": sums["mask_bce"] / counts["mask"],
    }


def selection_score(baseline: dict[str, float], candidate: dict[str, float]) -> float:
    weights = {"rgb_mae": 0.35, "high_motion_rgb_mae": 0.15, "temporal_delta_mae": 0.10, "highpass_mae": 0.15, "edge_mae": 0.10, "dark_region_rgb_mae": 0.15}
    value = sum(weight * candidate[name] / baseline[name] for name, weight in weights.items())
    value += 15.0 * max(candidate["rgb_mae"] / baseline["rgb_mae"] - 1.002, 0.0)
    value += 10.0 * max(candidate["high_motion_rgb_mae"] / baseline["high_motion_rgb_mae"] - 1.002, 0.0)
    return float(value)


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(output: Path, model, mean, std, manifest: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_save({"format": "track2-autoregressive-structure-unet-v3", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_autoregressive_structure_unet_config.npz", context_frames=5, prediction_frames=8, action_dim=14, working_resolution=256)
    temporary = output / f"training_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "training_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-autoregressive-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--structure-strength-learning-rate", type=float, help="Optional separate LR for the explicit RGB-coupling scalar.")
    parser.add_argument("--freeze-rgb-parent", action="store_true", help="Freeze the complete RGB parent and train only the mask channel and coupling scalar.")
    parser.add_argument("--coupling-warmup-steps", type=int, default=0, help="Train the structure mask alone before enabling RGB coupling.")
    parser.add_argument("--motion-weight", type=float, default=2.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--horizon-loss-power", type=float, default=1.0)
    parser.add_argument("--temporal-delta-weight", type=float, default=0.08)
    parser.add_argument("--texture-laplacian-weight", type=float, default=0.18)
    parser.add_argument("--mask-weight", type=float, default=0.02)
    parser.add_argument("--mask-edge-weight", type=float, default=0.10)
    parser.add_argument("--dark-rgb-weight", type=float, default=0.05)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=2.0)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--statistics-cache")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    split_path, windows = Path(args.split_manifest), Path(args.windows)
    split = json.loads(split_path.read_text())
    train = WindowDataset(windows, split["train_episodes"])
    validation = WindowDataset(windows, split["validation_episodes"])
    device = torch.device(args.device)
    mean, std, motion_scores = cached_training_statistics(train, Path(args.statistics_cache) if args.statistics_cache else None, windows, split_path)
    mean, std = mean.to(device), std.to(device)
    weights, sampling = high_motion_sampling(motion_scores, args.high_motion_threshold, args.high_motion_oversample_factor)
    sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    dev_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    torch.manual_seed(args.seed)
    model = OneStepActionSeparatedStructureUNet().to(device)
    parent = torch.load(Path(args.init_autoregressive_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
    if parent.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError("initial checkpoint is not an autoregressive U-Net")
    load_separated_rgb_parent(model, parent["state_dict"])
    strength_lr = args.structure_strength_learning_rate if args.structure_strength_learning_rate is not None else args.learning_rate
    if args.freeze_rgb_parent:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.structure_output.weight.requires_grad_(True)
        model.structure_output.bias.requires_grad_(True)
        model.structure_strength.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.structure_output.weight, model.structure_output.bias], "lr": args.learning_rate},
                {"params": [model.structure_strength], "lr": strength_lr},
            ],
            weight_decay=0.0,
        )
    else:
        ordinary = [parameter for name, parameter in model.named_parameters() if name != "structure_strength"]
        optimizer = torch.optim.AdamW(
            [{"params": ordinary, "lr": args.learning_rate}, {"params": [model.structure_strength], "lr": strength_lr}],
            weight_decay=1e-4,
        )
    config = {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "structure_strength_learning_rate": strength_lr, "freeze_rgb_parent": args.freeze_rgb_parent, "coupling_warmup_steps": args.coupling_warmup_steps, "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold, "horizon_loss_power": args.horizon_loss_power, "temporal_delta_weight": args.temporal_delta_weight, "texture_laplacian_weight": args.texture_laplacian_weight, "mask_weight": args.mask_weight, "mask_edge_weight": args.mask_edge_weight, "dark_rgb_weight": args.dark_rgb_weight, "high_motion_threshold": args.high_motion_threshold, "high_motion_sampling": sampling, "validation_sample_count": count, "seed": args.seed, "initial_checkpoint": str(Path(args.init_autoregressive_checkpoint).resolve())}
    output = Path(args.output)
    state_path = output / "training_state.pt"
    start, history, best = 0, [], float("inf")
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("format") != "track2-autoregressive-structure-training-v1" or state.get("config") != config:
            raise ValueError("structure training resume state does not match")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start, history, best = int(state["step"]), list(state["history"]), float(state["best_score"])
    baseline = evaluate(dev_loader, model, device, mean, std, args.high_motion_threshold)
    if not history:
        history.append({"step": 0, "selection_score": 1.0, **baseline})
        best = 1.0
        save_checkpoint(output / "best", model, mean, std, {"format": "track2-autoregressive-structure-unet-v3", **config, "checkpoint_step": 0, "best_selection_score": best, "validation": history, "baseline": baseline})
    iterator = iter(train_loader)
    model.train()
    for step in range(start + 1, args.steps + 1):
        try:
            context, history_actions, future, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        history_actions = ((history_actions.to(device, non_blocking=True) - mean) / std).float()
        future = ((future.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, logits = rollout(model, context, history_actions, future)
            previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
            reconstruction = reconstruction_loss(prediction, target, previous, args.motion_weight, args.motion_threshold, args.horizon_loss_power, args.temporal_delta_weight, args.texture_laplacian_weight, 4)
            auxiliary, components = structure_loss(prediction, logits, target, args.mask_weight, args.mask_edge_weight, args.dark_rgb_weight)
            loss = reconstruction + auxiliary
        loss.backward()
        if step <= args.coupling_warmup_steps and model.structure_strength.grad is not None:
            model.structure_strength.grad.zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()), **{name: float(value.detach()) for name, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(dev_loader, model, device, mean, std, args.high_motion_threshold)
            metric = selection_score(baseline, result)
            record = {"step": step, "selection_score": metric, **result}
            history.append(record)
            manifest = {"format": "track2-autoregressive-structure-unet-v3", **config, "checkpoint_step": step, "best_selection_score": min(best, metric), "validation": history, "baseline": baseline}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, manifest)
            if metric < best:
                best = metric
                manifest["best_checkpoint_step"] = step
                save_checkpoint(output / "best", model, mean, std, manifest)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_interval == 0 or step == args.steps:
            atomic_save({"format": "track2-autoregressive-structure-training-v1", "config": config, "step": step, "history": history, "best_score": best, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, state_path)


if __name__ == "__main__":
    main()
