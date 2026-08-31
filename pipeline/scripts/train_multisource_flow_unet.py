#!/usr/bin/env python3
"""Train the five-source Track 2 flow-fusion world model on train-only RAFT labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

# Keep the trainer executable both as ``python scripts/...`` and as an imported
# module in contract tests, without relying on a caller-provided PYTHONPATH.
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir))

from train_direct_flow_unet import (
    action_statistics,
    evaluate,
    frames_for_model,
    motion_sampler,
    reconstruction_loss,
)
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet


class MultiSourceWindowDataset(Dataset):
    """Windows with an 8-target by 5-source train-only RAFT tensor."""

    def __init__(self, directory: Path, episodes: list[int], flow_targets: Path | None = None, flow_resolution: int = 128) -> None:
        allowed = set(episodes)
        self.paths = [path for path in sorted(directory.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed]
        self.flow_targets = flow_targets
        self.flow_resolution = int(flow_resolution)
        if not self.paths:
            raise ValueError("no windows selected")
        if flow_targets is not None:
            missing = [path.name for path in self.paths if not (flow_targets / f"{path.stem}.npy").is_file()]
            if missing:
                raise ValueError(f"missing multi-source RAFT targets for {len(missing)} train windows; first={missing[0]}")

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
        expected = (8, 5, 2, self.flow_resolution, self.flow_resolution)
        if target.shape != expected:
            raise ValueError(f"invalid multi-source flow target shape for {path.name}: {target.shape}, expected {expected}")
        return (*values, torch.from_numpy(target.copy()))


def validate_flow_target_manifest(flow_targets: Path, windows: Path, split_manifest: Path, flow_resolution: int, expected_windows: int) -> dict:
    manifest_path = flow_targets / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing RAFT flow target manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "track2-raft-backward-flow-targets-v2":
        raise ValueError("multi-source trainer requires v2 RAFT target manifest")
    if manifest.get("split") != "train_episodes_only" or manifest.get("source_mode") != "all-context":
        raise ValueError("RAFT flow targets must be all-context labels generated from train episodes only")
    if int(manifest.get("source_count", -1)) != 5 or int(manifest.get("flow_resolution", -1)) != flow_resolution:
        raise ValueError("RAFT source count or resolution does not match multi-source trainer")
    if int(manifest.get("window_count", -1)) != expected_windows:
        raise ValueError("RAFT flow target window count does not match the train split")
    if Path(manifest.get("source_windows", "")).resolve() != windows.resolve():
        raise ValueError("RAFT flow targets were generated from a different windows directory")
    if Path(manifest.get("split_manifest", "")).resolve() != split_manifest.resolve():
        raise ValueError("RAFT flow targets were generated with a different episode split")
    return manifest


def validate_flow_target_files(dataset: MultiSourceWindowDataset) -> None:
    """Fail before CUDA allocation if any train label is missing or malformed."""
    if dataset.flow_targets is None:
        raise ValueError("multi-source training requires RAFT flow targets")
    expected = (8, 5, 2, dataset.flow_resolution, dataset.flow_resolution)
    for path in dataset.paths:
        value = np.load(dataset.flow_targets / f"{path.stem}.npy", mmap_mode="r", allow_pickle=False)
        if value.shape != expected or value.dtype != np.float16:
            raise ValueError(f"invalid multi-source RAFT target for {path.name}: shape={value.shape} dtype={value.dtype}")
        del value


def flow_supervision_loss(predicted_flow: torch.Tensor, target_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, steps, sources = predicted_flow.shape[:3]
    target_height, target_width = target_flow.shape[-2:]
    predicted_target = functional.interpolate(
        predicted_flow.flatten(0, 2), size=(target_height, target_width), mode="bilinear", align_corners=True
    ).reshape(batch, steps, sources, 2, target_height, target_width)
    scale = torch.tensor(
        (target_width / predicted_flow.shape[-1], target_height / predicted_flow.shape[-2]),
        dtype=predicted_target.dtype,
        device=predicted_target.device,
    ).view(1, 1, 1, 2, 1, 1)
    supervised = functional.smooth_l1_loss(predicted_target.mul(scale) / 16.0, target_flow / 16.0)
    smooth_x = (predicted_flow[..., 1:] - predicted_flow[..., :-1]).abs().mean()
    smooth_y = (predicted_flow[..., 1:, :] - predicted_flow[..., :-1, :]).abs().mean()
    return supervised, (smooth_x + smooth_y) / 2.0


def source_selection_loss(
    source_weight: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    teacher_flow: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teach fusion weights to choose the source that RAFT reconstructs best.

    Both the target image and RAFT flow are read only for training episodes.
    This directly supervises source selection at disocclusions, where a five-way
    RGB blend otherwise has no reason to prefer the visible historical view.
    """
    batch, steps, sources = teacher_flow.shape[:3]
    height, width = teacher_flow.shape[-2:]
    source_images = functional.interpolate(
        context.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=True
    ).reshape(batch, sources, 3, height, width)
    targets = functional.interpolate(
        target.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=True
    ).reshape(batch, steps, 3, height, width)
    # RAFT labels and resized sources share this exact 128px pixel coordinate system.
    teacher_warps = MultiSourceActionFlowUNet._warp(source_images, teacher_flow)
    source_error = (teacher_warps - targets[:, :, None]).abs().mean(dim=3)
    oracle_weight = torch.softmax(-source_error / temperature, dim=2)
    predicted_weight = functional.interpolate(
        source_weight.flatten(0, 2), size=(height, width), mode="bilinear", align_corners=True
    ).reshape(batch, steps, sources, 1, height, width).squeeze(3).clamp_min(1e-6)
    cross_entropy = -(oracle_weight * predicted_weight.log()).sum(dim=2).mean()
    oracle_best_error = (oracle_weight * source_error).sum(dim=2).mean()
    return cross_entropy, oracle_best_error


def save_checkpoint(output: Path, model, mean, std, metadata: dict, base_channels: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-multisource-flow-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(
        output / "track2_multisource_flow_unet_config.npz",
        context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8),
        working_resolution=np.asarray(256), serving_resolution=np.asarray(256), source_frames=np.asarray(5),
        base_channels=np.asarray(base_channels),
    )
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_training_state(
    output: Path,
    model,
    optimizer,
    scheduler,
    mean: torch.Tensor,
    std: torch.Tensor,
    step: int,
    best_mae: float,
    best_peak_mae: float,
    history: list[dict],
) -> None:
    """Atomically retain one resumable state beside the deployable best model."""
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".latest_training_state.tmp"
    torch.save(
        {
            "format": "track2-multisource-flow-unet-training-state-v2",
            "step": int(step),
            "best_mae": float(best_mae),
            "best_peak_mae": float(best_peak_mae),
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        },
        temporary,
    )
    temporary.replace(output / "latest_training_state.pt")


def better_validation_candidate(result: dict, best_peak_mae: float, best_mae: float) -> bool:
    """Rank deployable checkpoints by the strict gate before average quality.

    The formal acceptance criterion is the worst prediction frame over the
    held-out split. Mean MAE remains the deterministic tie breaker only.
    """
    return (float(result["max_window_frame_mae"]), float(result["mae"])) < (best_peak_mae, best_mae)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--flow-resolution", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--flow-loss-weight", type=float, default=0.8)
    parser.add_argument("--flow-smoothness-weight", type=float, default=0.002)
    parser.add_argument("--source-selection-loss-weight", type=float, default=0.15)
    parser.add_argument("--source-selection-temperature", type=float, default=0.025)
    parser.add_argument("--motion-weight", type=float, default=3.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=3.0)
    parser.add_argument("--accept-mae", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=0,
        help="Validation batches per checkpoint; 0 evaluates every held-out window (formal default).",
    )
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume the latest state in --output.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.base_channels, args.flow_resolution) < 1
        or args.base_channels % 8
        or args.accept_mae <= 0
        or args.source_selection_loss_weight < 0
        or args.source_selection_temperature <= 0
        or args.validation_interval < 1
        or args.validation_batches < 0
    ):
        raise SystemExit("invalid training hyperparameters")

    torch.manual_seed(args.seed)
    split_manifest, windows, targets = Path(args.split_manifest), Path(args.windows), Path(args.flow_targets)
    split = json.loads(split_manifest.read_text())
    train = MultiSourceWindowDataset(windows, split["train_episodes"], targets, args.flow_resolution)
    validation = MultiSourceWindowDataset(windows, split["validation_episodes"])
    flow_manifest = validate_flow_target_manifest(targets, windows, split_manifest, args.flow_resolution, len(train))
    validate_flow_target_files(train)
    device = torch.device(args.device)
    mean, std = action_statistics(train)
    mean, std = mean.to(device), std.to(device)
    sampler, sampling = motion_sampler(train, args.high_motion_threshold, args.high_motion_oversample_factor, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    count = len(validation) if args.validation_batches == 0 else min(len(validation), args.validation_batches * args.batch_size)
    indices = list(range(len(validation))) if count == len(validation) else np.linspace(0, len(validation) - 1, count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    output = Path(args.output)
    model = MultiSourceActionFlowUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    history, best_mae, best_peak_mae, start_step = [], float("inf"), float("inf"), 0
    state_path = output / "latest_training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("format") not in {
            "track2-multisource-flow-unet-training-state-v1",
            "track2-multisource-flow-unet-training-state-v2",
        }:
            raise ValueError("unsupported multi-source training-state format")
        start_step = int(state["step"])
        if not 0 <= start_step < args.steps:
            raise ValueError("resume state step must be less than --steps")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        saved_mean, saved_std = state["mean"].to(device), state["std"].to(device)
        if not torch.allclose(mean, saved_mean) or not torch.allclose(std, saved_std):
            raise ValueError("resume state action normalization does not match current train split")
        history, best_mae = list(state["history"]), float(state["best_mae"])
        best_peak_mae = float(state.get("best_peak_mae", float("inf")))
        print(
            json.dumps(
                {"status": "resumed", "step": start_step, "best_mae": best_mae, "best_peak_mae": best_peak_mae}
            ),
            flush=True,
        )
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
            prediction, flow, source_weight = model(context, ((actions - mean) / std).float(), return_flow=True)
            prediction = prediction.clamp(0.0, 1.0)
            image_loss = reconstruction_loss(prediction, target, context[:, -1], args.motion_weight, args.motion_threshold)
            flow_loss, smoothness = flow_supervision_loss(flow.float(), teacher_flow)
            selection_loss, oracle_source_error = source_selection_loss(
                source_weight.float(), context.float(), target.float(), teacher_flow, args.source_selection_temperature
            )
            loss = (
                image_loss
                + args.flow_loss_weight * flow_loss
                + args.flow_smoothness_weight * smoothness
                + args.source_selection_loss_weight * selection_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "image_loss": float(image_loss.detach().cpu()), "flow_loss": float(flow_loss.detach().cpu()), "flow_smoothness": float(smoothness.detach().cpu()), "source_selection_loss": float(selection_loss.detach().cpu()), "oracle_source_error": float(oracle_source_error.detach().cpu()), "mae": float(functional.l1_loss(prediction.float(), target).detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.accept_mae)
            model.train()
            result["step"] = step
            history.append(result)
            metadata = {
                "backend": "multisource-flow-unet", "format": "track2-multisource-flow-unet-v1", "windows": str(windows.resolve()),
                "split_manifest": str(split_manifest.resolve()), "flow_targets": str(targets.resolve()),
                "flow_target_manifest": str((targets / "manifest.json").resolve()), "flow_target_split": "train_episodes_only",
                "flow_target_resolution": args.flow_resolution, "flow_target_raft_weights": flow_manifest.get("raft_weights"),
                "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": count,
                "validation_is_full_held_out_split": count == len(validation),
                "context_frames": 5, "history_actions": 4, "future_actions": 8, "target_frames": 8, "action_dim": 14,
                "working_resolution": 256, "serving_resolution": 256, "source_frames": 5, "base_channels": args.base_channels,
                "training_steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "flow_loss_weight": args.flow_loss_weight, "flow_smoothness_weight": args.flow_smoothness_weight,
                "source_selection_loss_weight": args.source_selection_loss_weight,
                "source_selection_temperature": args.source_selection_temperature,
                "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold, "high_motion_sampling": sampling,
                "accept_mae_0_255": args.accept_mae, "checkpoint_step": step, "validation": history,
                "checkpoint_selection": "min(max_window_frame_mae, mae)",
                "best_validation_mae": best_mae,
                "best_validation_max_window_frame_mae": best_peak_mae,
            }
            if args.save_all_checkpoints:
                save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, metadata, args.base_channels)
            if better_validation_candidate(result, best_peak_mae, best_mae):
                best_mae = result["mae"]
                best_peak_mae = result["max_window_frame_mae"]
                metadata["best_checkpoint_step"] = step
                metadata["best_validation_mae"] = best_mae
                metadata["best_validation_max_window_frame_mae"] = best_peak_mae
                save_checkpoint(output / "best", model, mean, std, metadata, args.base_channels)
                # Keep a lightweight run manifest at the output root. The only
                # runnable model files live in best/ to avoid duplicate weights.
                (output / "training_manifest.json").parent.mkdir(parents=True, exist_ok=True)
                (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            save_training_state(output, model, optimizer, scheduler, mean, std, step, best_mae, best_peak_mae, history)
            if step == args.steps:
                completion = {
                    "format": "track2-multisource-flow-unet-completion-v1",
                    "completed_steps": step,
                    "best_validation_mae": best_mae,
                    "best_validation_max_window_frame_mae": best_peak_mae,
                    "best_checkpoint": str((output / "best").resolve()),
                }
                (output / "training_complete.json").write_text(json.dumps(completion, indent=2) + "\n")
            print(json.dumps({"step": step, "validation": result, "best_mae": best_mae}), flush=True)


if __name__ == "__main__":
    main()
