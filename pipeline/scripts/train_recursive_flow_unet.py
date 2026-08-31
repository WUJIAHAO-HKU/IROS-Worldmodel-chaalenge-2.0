#!/usr/bin/env python3
"""Fine-tune a Direct Flow checkpoint as an eight-step recursive world model."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.recursive_flow_unet import RecursiveActionFlowUNet


class WindowDataset(Dataset):
    def __init__(self, directory: Path, episodes: list[int], limit: int | None = None) -> None:
        allowed = set(episodes)
        self.paths = [
            path
            for path in sorted(directory.glob("episode*_*.npz"))
            if int(path.name.split("_")[0][7:]) in allowed
        ]
        if limit is not None:
            self.paths = self.paths[:limit]
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


def load_direct_warm_start(
    checkpoint: Path, model: RecursiveActionFlowUNet, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    config_path = checkpoint / "track2_direct_flow_unet_config.npz"
    if not config_path.is_file():
        raise ValueError(f"missing Direct Flow config: {config_path}")
    config = np.load(config_path, allow_pickle=False)
    expected = (5, 14, 8, model.base_channels)
    actual = tuple(int(config[key]) for key in ("context_frames", "action_dim", "prediction_frames", "base_channels"))
    if actual != expected:
        raise ValueError(f"Direct Flow warm start profile mismatch: expected {expected}, got {actual}")
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-direct-flow-unet-v1":
        raise ValueError("warm start is not a Direct Flow v1 checkpoint")
    model.load_state_dict(state["state_dict"], strict=True)
    normalization = np.load(checkpoint / "action_normalization.npz", allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], dtype=np.float32)).to(device)
    if mean.shape != (14,) or std.shape != (14,) or bool((std <= 0).any()):
        raise ValueError("invalid Direct Flow action normalization")
    manifest_path = checkpoint / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return mean, std, manifest


def step_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous_target: torch.Tensor,
    motion_weight: float,
    motion_threshold: float,
) -> torch.Tensor:
    changed = (target - previous_target).abs().mean(dim=1, keepdim=True) >= motion_threshold
    weights = 1.0 + (motion_weight - 1.0) * changed.to(target.dtype)
    pixel = (weights * (prediction - target).abs()).mean()
    pixel = pixel + 0.03 * (weights * (prediction - target).square()).mean()
    coarse = functional.l1_loss(functional.avg_pool2d(prediction, 2), functional.avg_pool2d(target, 2))
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    delta = functional.l1_loss(prediction - previous_target, target - previous_target)
    return pixel + 0.15 * coarse + 0.12 * (edge_x + edge_y) / 2.0 + 0.25 * delta


def rollout(model, context, history, future):
    predictions = []
    for action in future.unbind(dim=1):
        frame = model(context, torch.cat((history, action[:, None]), dim=1)).clamp(0.0, 1.0)
        predictions.append(frame)
        context = torch.cat((context[:, 1:], frame[:, None]), dim=1)
        history = torch.cat((history[:, 1:], action[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def evaluate(loader, model, device, mean, std, accept_mae: float) -> dict:
    model.eval()
    errors = []
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for context, history, future, target in loader:
            context = frames_for_model(context).to(device, non_blocking=True)
            target = frames_for_model(target).to(device, non_blocking=True)
            history = ((history.to(device, non_blocking=True) - mean) / std).float()
            future = ((future.to(device, non_blocking=True) - mean) / std).float()
            prediction = rollout(model, context, history, future)
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


def save_checkpoint(output: Path, model, mean, std, metadata: dict, base_channels: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-recursive-flow-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(
        output / "track2_recursive_flow_unet_config.npz",
        context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8),
        working_resolution=np.asarray(256), serving_resolution=np.asarray(256),
        base_channels=np.asarray(base_channels),
    )
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_training_state(output: Path, model, optimizer, scheduler, step: int, target_steps: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "training_state.pt.tmp"
    torch.save(
        {
            "format": "track2-recursive-flow-training-state-v1",
            "step": step,
            "target_steps": target_steps,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary,
    )
    temporary.replace(output / "training_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--motion-weight", type=float, default=4.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--flow-smoothness-weight", type=float, default=0.001)
    parser.add_argument("--accept-mae", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.base_channels, args.validation_interval, args.checkpoint_interval) < 1 or args.base_channels % 8:
        raise SystemExit("steps, batch size, validation interval, and a multiple-of-8 base channel count must be positive")
    if args.learning_rate <= 0 or args.accept_mae <= 0 or args.motion_weight < 1:
        raise SystemExit("learning rate and accept MAE must be positive; motion weight must be at least one")

    torch.manual_seed(args.seed)
    split_path, windows = Path(args.split_manifest), Path(args.windows)
    split = json.loads(split_path.read_text())
    train = WindowDataset(windows, split["train_episodes"], args.max_train_windows)
    validation = WindowDataset(windows, split["validation_episodes"], args.max_validation_windows)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = RecursiveActionFlowUNet(args.base_channels).to(device)
    warm_start = Path(args.warm_start)
    mean, std, source_manifest = load_direct_warm_start(warm_start, model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05
    )
    output, history, best_mae = Path(args.output), [], float("inf")
    start_step = 0
    state_path = output / "training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != "track2-recursive-flow-training-state-v1":
            raise ValueError("unsupported recursive Flow training state")
        start_step = int(state["step"])
        if int(state.get("target_steps", -1)) != args.steps:
            raise ValueError("resume state was created for a different --steps target")
        if not 0 <= start_step <= args.steps:
            raise ValueError("resume step is outside the requested training horizon")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        manifest_path = output / "training_manifest.json"
        if manifest_path.is_file():
            prior = json.loads(manifest_path.read_text())
            history = prior.get("validation", [])
            best_mae = float(prior.get("best_validation_mae", float("inf")))
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
            context, history_actions, future_actions, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future_actions, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        history_actions = ((history_actions.to(device, non_blocking=True) - mean) / std).float()
        future_actions = ((future_actions.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        losses, errors = [], []
        previous_target = context[:, -1]

        # Each horizon backpropagates independently. Detaching the recurrent frame
        # keeps eight-step training within one-GPU memory while still training on
        # the model's own accumulated rollout state rather than teacher forcing.
        for horizon, action in enumerate(future_actions.unbind(dim=1)):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, flow = model(
                    context, torch.cat((history_actions, action[:, None]), dim=1), return_flow=True
                )
                prediction = prediction.clamp(0.0, 1.0)
                image_loss = step_reconstruction_loss(
                    prediction, target[:, horizon], previous_target,
                    args.motion_weight, args.motion_threshold,
                )
                smooth_x = (flow[..., 1:] - flow[..., :-1]).abs().mean()
                smooth_y = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
                loss = image_loss + args.flow_smoothness_weight * (smooth_x + smooth_y) / 2.0
            (loss / 8.0).backward()
            losses.append(float(loss.detach().cpu()))
            errors.append(float(functional.l1_loss(prediction.float(), target[:, horizon]).detach().cpu()))
            frame = prediction.detach()
            context = torch.cat((context[:, 1:], frame[:, None]), dim=1)
            history_actions = torch.cat((history_actions[:, 1:], action[:, None]), dim=1)
            previous_target = target[:, horizon]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(np.mean(losses)), "mae": float(np.mean(errors)), "mae_by_prediction_frame": errors, "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.accept_mae)
            model.train()
            result["step"] = step
            history.append(result)
            metadata = {
                "backend": "recursive-flow-unet", "format": "track2-recursive-flow-unet-v1",
                "windows": str(windows.resolve()), "split_manifest": str(split_path.resolve()),
                "warm_start": str(warm_start.resolve()), "warm_start_format": source_manifest.get("format"),
                "warm_start_checkpoint_step": source_manifest.get("checkpoint_step"),
                "train_window_count": len(train), "validation_window_count": len(validation),
                "validation_sample_count": count, "context_frames": 5, "history_actions": 4,
                "future_actions": 8, "target_frames": 8, "action_dim": 14,
                "working_resolution": 256, "serving_resolution": 256, "base_channels": args.base_channels,
                "training_steps": args.steps, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "motion_weight": args.motion_weight,
                "motion_threshold": args.motion_threshold, "flow_smoothness_weight": args.flow_smoothness_weight,
                "rollout_training": "self_prediction_detached_between_horizons",
                "accept_mae_0_255": args.accept_mae, "checkpoint_step": step,
                "validation": history, "best_validation_mae": min(best_mae, result["mae"]),
            }
            if result["mae"] < best_mae:
                best_mae = result["mae"]
                metadata["best_checkpoint_step"] = step
                save_checkpoint(output, model, mean, std, metadata, args.base_channels)
                save_checkpoint(output / "best", model, mean, std, metadata, args.base_channels)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps or stop_requested:
            save_training_state(output, model, optimizer, scheduler, step, args.steps)
        if stop_requested:
            print(json.dumps({"step": step, "status": "checkpointed_for_gpu_yield"}), flush=True)
            return


if __name__ == "__main__":
    main()
