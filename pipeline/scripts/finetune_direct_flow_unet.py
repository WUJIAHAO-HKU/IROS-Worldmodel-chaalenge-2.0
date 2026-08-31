#!/usr/bin/env python3
"""Resume a Direct Flow checkpoint with long-horizon/high-motion emphasis."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset

from train_direct_flow_unet import (
    WindowDataset,
    frames_for_model,
    motion_sampler,
    save_checkpoint,
    validate_flow_target_manifest,
)
from wam_pipeline.direct_flow_unet import DirectActionFlowUNet


def horizon_weights(steps: int, power: float, device, dtype) -> torch.Tensor:
    weights = torch.arange(1, steps + 1, device=device, dtype=dtype).div(steps).pow(power)
    return weights.div(weights.mean()).view(1, steps, 1, 1, 1)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    last: torch.Tensor,
    motion_weight: float,
    motion_threshold: float,
    horizon_power: float,
) -> torch.Tensor:
    previous = torch.cat((last[:, None], target[:, :-1]), dim=1)
    changed = (target - previous).abs().mean(dim=2, keepdim=True) >= motion_threshold
    weights = 1.0 + (motion_weight - 1.0) * changed.to(target.dtype)
    weights = weights * horizon_weights(target.shape[1], horizon_power, target.device, target.dtype)
    pixel = (weights * (prediction - target).abs()).mean()
    pixel = pixel + 0.03 * (weights * (prediction - target).square()).mean()
    batch, steps = prediction.shape[:2]
    pred_flat, target_flat = prediction.flatten(0, 1), target.flatten(0, 1)
    coarse_per_pixel = (
        functional.avg_pool2d(pred_flat, 2) - functional.avg_pool2d(target_flat, 2)
    ).abs().reshape(batch, steps, 3, target.shape[-2] // 2, target.shape[-1] // 2)
    coarse = (coarse_per_pixel * horizon_weights(steps, horizon_power, target.device, target.dtype)).mean()
    edge_x = (prediction[..., 1:] - prediction[..., :-1] - target[..., 1:] + target[..., :-1]).abs()
    edge_y = (prediction[..., 1:, :] - prediction[..., :-1, :] - target[..., 1:, :] + target[..., :-1, :]).abs()
    hweights = horizon_weights(steps, horizon_power, target.device, target.dtype)
    edges = (edge_x * hweights).mean().add((edge_y * hweights).mean()).div(2.0)
    delta = ((prediction - last[:, None] - target + last[:, None]).abs() * hweights).mean()
    return pixel + 0.15 * coarse + 0.12 * edges + 0.25 * delta


def flow_loss(
    predicted: torch.Tensor, target: torch.Tensor, horizon_power: float
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps = predicted.shape[:2]
    height, width = target.shape[-2:]
    resized = functional.interpolate(
        predicted.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=True
    ).reshape(batch, steps, 2, height, width)
    scale = torch.tensor(
        (width / predicted.shape[-1], height / predicted.shape[-2]),
        dtype=resized.dtype, device=resized.device,
    ).view(1, 1, 2, 1, 1)
    weights = horizon_weights(steps, horizon_power, resized.device, resized.dtype)
    supervised = functional.smooth_l1_loss(resized.mul(scale) / 16.0, target / 16.0, reduction="none")
    supervised = (supervised * weights).mean()
    smooth_x = ((predicted[..., 1:] - predicted[..., :-1]).abs() * weights).mean()
    smooth_y = ((predicted[..., 1:, :] - predicted[..., :-1, :]).abs() * weights).mean()
    return supervised, (smooth_x + smooth_y) / 2.0


def evaluate(loader, model, device, mean, std, high_motion_threshold: float, accept_mae: float) -> dict:
    model.eval()
    errors, motion_scores = [], []
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for context, history, future, target in loader:
            context = frames_for_model(context).to(device, non_blocking=True)
            target = frames_for_model(target).to(device, non_blocking=True)
            actions = torch.cat((history, future), dim=1).to(device, non_blocking=True)
            prediction = model(context, ((actions - mean) / std).float()).clamp(0.0, 1.0)
            errors.append((prediction.float() - target.float()).abs().mean(dim=(2, 3, 4)).cpu())
            previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
            motion_scores.append((target.float() - previous.float()).abs().mean(dim=(1, 2, 3, 4)).cpu())
    error, motion = torch.cat(errors), torch.cat(motion_scores)
    high = motion >= high_motion_threshold
    window_mean, window_peak = error.mean(dim=1), error.max(dim=1).values
    return {
        "mae": float(error.mean()),
        "mae_by_prediction_frame": [float(value) for value in error.mean(dim=0)],
        "high_motion_threshold": float(high_motion_threshold),
        "high_motion_sample_count": int(high.sum()),
        "high_motion_mae": float(error[high].mean()) if bool(high.any()) else None,
        "max_window_mean_mae": float(window_mean.max()),
        "max_window_frame_mae": float(window_peak.max()),
        "windows_passing_mean": int((window_mean * 255.0 < accept_mae).sum()),
        "windows_passing_all_frames": int((window_peak * 255.0 < accept_mae).sum()),
        "window_count": int(len(error)),
        "accept_mae_0_255": float(accept_mae),
        "all_windows_and_frames_pass": bool((window_peak * 255.0 < accept_mae).all()),
    }


def load_initialization(checkpoint: Path, model: DirectActionFlowUNet, device: torch.device):
    config = np.load(checkpoint / "track2_direct_flow_unet_config.npz", allow_pickle=False)
    expected = (5, 14, 8, model.base_channels)
    actual = tuple(int(config[key]) for key in ("context_frames", "action_dim", "prediction_frames", "base_channels"))
    if actual != expected:
        raise ValueError(f"Direct Flow initialization mismatch: expected {expected}, got {actual}")
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-direct-flow-unet-v1":
        raise ValueError("initialization is not a Direct Flow v1 checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], dtype=np.float32)).to(device)
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    return mean, std, manifest


def save_training_state(output: Path, model, optimizer, scheduler, step: int, target_steps: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "training_state.pt.tmp"
    torch.save({
        "format": "track2-direct-flow-finetune-state-v1", "step": step, "target_steps": target_steps,
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }, temporary)
    temporary.replace(output / "training_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--flow-resolution", type=int, default=128)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--flow-loss-weight", type=float, default=0.8)
    parser.add_argument("--flow-smoothness-weight", type=float, default=0.001)
    parser.add_argument("--motion-weight", type=float, default=5.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--horizon-loss-power", type=float, default=0.5)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=5.0)
    parser.add_argument("--high-motion-selection-weight", type=float, default=0.5)
    parser.add_argument("--accept-mae", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.base_channels, args.validation_interval, args.checkpoint_interval) < 1:
        raise SystemExit("step, batch, channel, validation, and checkpoint counts must be positive")
    if args.base_channels % 8 or args.horizon_loss_power < 0 or not 0 <= args.high_motion_selection_weight <= 1:
        raise SystemExit("base channels must divide by 8; horizon power and selection weight are invalid")

    torch.manual_seed(args.seed)
    windows, split_path, flow_targets = Path(args.windows), Path(args.split_manifest), Path(args.flow_targets)
    split = json.loads(split_path.read_text())
    train = WindowDataset(windows, split["train_episodes"], flow_targets, args.flow_resolution)
    validation = WindowDataset(windows, split["validation_episodes"])
    flow_manifest = validate_flow_target_manifest(
        flow_targets, windows, split_path, args.flow_resolution, len(train)
    )
    sampler, sampling = motion_sampler(
        train, args.high_motion_threshold, args.high_motion_oversample_factor, args.seed
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = DirectActionFlowUNet(args.base_channels).to(device)
    initialization = Path(args.init_checkpoint)
    mean, std, source_manifest = load_initialization(initialization, model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1
    )
    output, start_step, best_metric, history = Path(args.output), 0, float("inf"), []
    state_path = output / "training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != "track2-direct-flow-finetune-state-v1" or int(state.get("target_steps", -1)) != args.steps:
            raise ValueError("incompatible Direct Flow fine-tune resume state")
        start_step = int(state["step"])
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        manifest_path = output / "training_manifest.json"
        if manifest_path.is_file():
            prior = json.loads(manifest_path.read_text())
            history = prior.get("validation", [])
            best_metric = float(prior.get("best_selection_metric", float("inf")))
        print(json.dumps({"resumed_from_step": start_step, "target_step": args.steps}), flush=True)

    stop_requested = False
    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"signal": signum, "status": "checkpoint_requested"}), flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    iterator = iter(train_loader)

    for step in range(start_step + 1, args.steps + 1):
        try:
            context, history_actions, future, target, teacher_flow = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future, target, teacher_flow = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        teacher_flow = teacher_flow.to(device, non_blocking=True)
        actions = torch.cat((history_actions, future), dim=1).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, flow = model(context, ((actions - mean) / std).float(), return_flow=True)
            prediction = prediction.clamp(0.0, 1.0)
            image_loss = reconstruction_loss(
                prediction, target, context[:, -1], args.motion_weight,
                args.motion_threshold, args.horizon_loss_power,
            )
            supervised_flow, smoothness = flow_loss(flow.float(), teacher_flow, args.horizon_loss_power)
            loss = image_loss + args.flow_loss_weight * supervised_flow + args.flow_smoothness_weight * smoothness
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "image_loss": float(image_loss.detach().cpu()), "flow_loss": float(supervised_flow.detach().cpu()), "mae": float(functional.l1_loss(prediction.float(), target).detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)

        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.high_motion_threshold, args.accept_mae)
            model.train()
            result["step"] = step
            high_mae = result["high_motion_mae"]
            selection_metric = result["mae"] if high_mae is None else (
                (1.0 - args.high_motion_selection_weight) * result["mae"]
                + args.high_motion_selection_weight * high_mae
            )
            result["selection_metric"] = selection_metric
            history.append(result)
            metadata = {
                "backend": "direct-flow-unet", "format": "track2-direct-flow-unet-v1",
                "finetune_format": "track2-direct-flow-long-horizon-finetune-v1",
                "windows": str(windows.resolve()), "split_manifest": str(split_path.resolve()),
                "flow_targets": str(flow_targets.resolve()), "flow_target_manifest": str((flow_targets / "manifest.json").resolve()),
                "flow_target_split": "train_episodes_only", "flow_target_resolution": args.flow_resolution,
                "flow_target_raft_weights": flow_manifest.get("raft_weights"),
                "initialization_checkpoint": str(initialization.resolve()),
                "initialization_checkpoint_step": source_manifest.get("checkpoint_step"),
                "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": count,
                "context_frames": 5, "history_actions": 4, "future_actions": 8, "target_frames": 8, "action_dim": 14,
                "working_resolution": 256, "serving_resolution": 256, "base_channels": args.base_channels,
                "training_steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "flow_loss_weight": args.flow_loss_weight, "flow_smoothness_weight": args.flow_smoothness_weight,
                "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold,
                "horizon_loss_power": args.horizon_loss_power, "high_motion_sampling": sampling,
                "high_motion_selection_weight": args.high_motion_selection_weight,
                "accept_mae_0_255": args.accept_mae, "checkpoint_step": step,
                "validation": history, "best_selection_metric": min(best_metric, selection_metric),
            }
            if selection_metric < best_metric:
                best_metric = selection_metric
                metadata["best_checkpoint_step"] = step
                metadata["best_validation_mae"] = result["mae"]
                metadata["best_high_motion_validation_mae"] = result["high_motion_mae"]
                save_checkpoint(output, model, mean, std, metadata, args.base_channels)
                save_checkpoint(output / "best", model, mean, std, metadata, args.base_channels)
            print(json.dumps({"step": step, "validation": result, "best_selection_metric": best_metric}), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps or stop_requested:
            save_training_state(output, model, optimizer, scheduler, step, args.steps)
        if stop_requested:
            print(json.dumps({"step": step, "status": "checkpointed_for_gpu_yield"}), flush=True)
            return


if __name__ == "__main__":
    main()
