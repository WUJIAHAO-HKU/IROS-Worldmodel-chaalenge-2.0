#!/usr/bin/env python3
"""Train a direct action-conditioned flow world model using train-only RAFT targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from wam_pipeline.direct_flow_unet import DirectActionFlowUNet


class WindowDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        episodes: list[int],
        flow_targets: Path | None = None,
        flow_resolution: int = 64,
    ) -> None:
        allowed = set(episodes)
        self.paths = [path for path in sorted(directory.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed]
        self.flow_targets = flow_targets
        self.flow_resolution = flow_resolution
        if not self.paths:
            raise ValueError("no windows selected")
        if flow_targets is not None:
            missing = [path.name for path in self.paths if not (flow_targets / f"{path.stem}.npy").is_file()]
            if missing:
                raise ValueError(f"missing RAFT flow targets for {len(missing)} train windows; first={missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            values = (
                torch.from_numpy(data["context_frames"].copy()),
                torch.from_numpy(data["history_actions"].copy()),
                torch.from_numpy(data["future_actions"].copy()),
                torch.from_numpy(data["target_frames"].copy()),
            )
        if self.flow_targets is None:
            return values
        target = np.load(self.flow_targets / f"{path.stem}.npy", allow_pickle=False).astype(np.float32, copy=False)
        if target.shape != (8, 2, self.flow_resolution, self.flow_resolution):
            raise ValueError(f"invalid flow target shape for {path.name}: {target.shape}")
        return (*values, torch.from_numpy(target.copy()))


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


def validate_flow_target_manifest(
    flow_targets: Path, windows: Path, split_manifest: Path, flow_resolution: int, expected_windows: int
) -> dict:
    manifest_path = flow_targets / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing RAFT flow target manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "track2-raft-backward-flow-targets-v1":
        raise ValueError("unsupported RAFT flow target manifest format")
    if manifest.get("split") != "train_episodes_only":
        raise ValueError("RAFT flow targets must be generated from train episodes only")
    if int(manifest.get("flow_resolution", -1)) != flow_resolution:
        raise ValueError("RAFT flow target resolution does not match --flow-resolution")
    if int(manifest.get("window_count", -1)) != expected_windows:
        raise ValueError("RAFT flow target window count does not match the train split")
    if Path(manifest.get("source_windows", "")).resolve() != windows.resolve():
        raise ValueError("RAFT flow targets were generated from a different windows directory")
    if Path(manifest.get("split_manifest", "")).resolve() != split_manifest.resolve():
        raise ValueError("RAFT flow targets were generated with a different episode split")
    return manifest


def motion_sampler(dataset: WindowDataset, threshold: float, factor: float, seed: int) -> tuple[WeightedRandomSampler, dict]:
    scores = []
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            target = data["target_frames"].astype(np.float32) / 255.0
            previous = np.concatenate((data["context_frames"][-1:].astype(np.float32) / 255.0, target[:-1]), axis=0)
        scores.append(float(np.abs(target - previous).mean()))
    scores = np.asarray(scores, dtype=np.float64)
    high = scores >= threshold
    weights = np.ones(len(scores), dtype=np.float64)
    weights[high] = factor
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(dataset), replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    return sampler, {
        "threshold": float(threshold), "oversample_factor": float(factor), "high_motion_window_count": int(high.sum()),
        "high_motion_fraction": float(high.mean()), "expected_high_motion_draw_fraction": float(weights[high].sum() / weights.sum()) if high.any() else 0.0,
    }


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, last: torch.Tensor, motion_weight: float, motion_threshold: float) -> torch.Tensor:
    previous = torch.cat((last[:, None], target[:, :-1]), dim=1)
    changed = (target - previous).abs().mean(dim=2, keepdim=True) >= motion_threshold
    weights = 1.0 + (motion_weight - 1.0) * changed.to(target.dtype)
    pixel = (weights * (prediction - target).abs()).mean() + 0.03 * (weights * (prediction - target).square()).mean()
    pred_flat, target_flat = prediction.flatten(0, 1), target.flatten(0, 1)
    coarse = functional.l1_loss(functional.avg_pool2d(pred_flat, 2), functional.avg_pool2d(target_flat, 2))
    edge_x = functional.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    edge_y = functional.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    delta = functional.l1_loss(prediction - last[:, None], target - last[:, None])
    return pixel + 0.15 * coarse + 0.12 * (edge_x + edge_y) / 2.0 + 0.25 * delta


def flow_supervision_loss(predicted_flow: torch.Tensor, target_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps = predicted_flow.shape[:2]
    target_height, target_width = target_flow.shape[-2:]
    predicted_target = functional.interpolate(
        predicted_flow.flatten(0, 1), size=(target_height, target_width), mode="bilinear", align_corners=True
    ).reshape(batch, steps, 2, target_height, target_width)
    # RAFT labels use the pixel coordinates of their own resolution. Model flow is 256px.
    scale = torch.tensor(
        (target_width / predicted_flow.shape[-1], target_height / predicted_flow.shape[-2]),
        dtype=predicted_target.dtype,
        device=predicted_target.device,
    ).view(1, 1, 2, 1, 1)
    supervised = functional.smooth_l1_loss(predicted_target.mul(scale) / 16.0, target_flow / 16.0)
    smooth_x = (predicted_flow[..., 1:] - predicted_flow[..., :-1]).abs().mean()
    smooth_y = (predicted_flow[..., 1:, :] - predicted_flow[..., :-1, :]).abs().mean()
    return supervised, (smooth_x + smooth_y) / 2.0


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
        "mae": float(error.mean()), "mae_by_prediction_frame": [float(value) for value in error.mean(dim=0)],
        "max_window_mean_mae": float(window_mean.max()), "max_window_frame_mae": float(window_peak.max()),
        "windows_passing_mean": int((window_mean * 255.0 < accept_mae).sum()),
        "windows_passing_all_frames": int((window_peak * 255.0 < accept_mae).sum()), "window_count": int(len(error)),
        "accept_mae_0_255": float(accept_mae), "all_windows_and_frames_pass": bool((window_peak * 255.0 < accept_mae).all()),
    }


def save_checkpoint(output: Path, model, mean, std, metadata: dict, base_channels: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-direct-flow-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_direct_flow_unet_config.npz", context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8), working_resolution=np.asarray(256), serving_resolution=np.asarray(256), base_channels=np.asarray(base_channels))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--flow-resolution", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--flow-loss-weight", type=float, default=0.8)
    parser.add_argument("--flow-smoothness-weight", type=float, default=0.002)
    parser.add_argument("--motion-weight", type=float, default=3.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=3.0)
    parser.add_argument("--accept-mae", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=128)
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Keep every periodic checkpoint; by default only the best runnable checkpoint is retained.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.base_channels, args.flow_resolution) < 1 or args.base_channels % 8 or args.accept_mae <= 0:
        raise SystemExit("steps, batch size, flow resolution, accept MAE, and a multiple-of-8 base channel count must be positive")
    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    train = WindowDataset(
        Path(args.windows), split["train_episodes"], Path(args.flow_targets), args.flow_resolution
    )
    validation = WindowDataset(Path(args.windows), split["validation_episodes"])
    flow_manifest = validate_flow_target_manifest(
        Path(args.flow_targets), Path(args.windows), Path(args.split_manifest), args.flow_resolution, len(train)
    )
    device = torch.device(args.device)
    mean, std = action_statistics(train)
    mean, std = mean.to(device), std.to(device)
    sampler, sampling = motion_sampler(train, args.high_motion_threshold, args.high_motion_oversample_factor, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    count = min(len(validation), args.validation_batches * args.batch_size)
    indices = np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = DirectActionFlowUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    output, history, best_mae = Path(args.output), [], float("inf")
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
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
            image_loss = reconstruction_loss(prediction, target, context[:, -1], args.motion_weight, args.motion_threshold)
            flow_loss, smoothness = flow_supervision_loss(flow.float(), teacher_flow)
            loss = image_loss + args.flow_loss_weight * flow_loss + args.flow_smoothness_weight * smoothness
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "image_loss": float(image_loss.detach().cpu()), "flow_loss": float(flow_loss.detach().cpu()), "flow_smoothness": float(smoothness.detach().cpu()), "mae": float(functional.l1_loss(prediction.float(), target).detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.accept_mae)
            model.train()
            result["step"] = step
            history.append(result)
            metadata = {
                "backend": "direct-flow-unet", "format": "track2-direct-flow-unet-v1", "windows": str(Path(args.windows).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()), "flow_targets": str(Path(args.flow_targets).resolve()),
                "flow_target_manifest": str((Path(args.flow_targets) / "manifest.json").resolve()), "flow_target_split": "train_episodes_only", "flow_target_resolution": args.flow_resolution, "flow_target_raft_weights": flow_manifest.get("raft_weights"),
                "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": count,
                "context_frames": 5, "history_actions": 4, "future_actions": 8, "target_frames": 8, "action_dim": 14,
                "working_resolution": 256, "serving_resolution": 256, "base_channels": args.base_channels, "training_steps": args.steps, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "flow_loss_weight": args.flow_loss_weight, "flow_smoothness_weight": args.flow_smoothness_weight,
                "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold, "high_motion_sampling": sampling,
                "accept_mae_0_255": args.accept_mae, "checkpoint_step": step, "validation": history, "best_validation_mae": min(best_mae, result["mae"]),
            }
            if args.save_all_checkpoints:
                save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, metadata, args.base_channels)
            if result["mae"] < best_mae:
                best_mae = result["mae"]
                metadata["best_checkpoint_step"] = step
                save_checkpoint(output, model, mean, std, metadata, args.base_channels)
                save_checkpoint(output / "best", model, mean, std, metadata, args.base_channels)
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
