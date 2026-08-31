#!/usr/bin/env python3
"""Distill RAFT into an AR-conditioned texture reprojection/structure parent."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from wam_pipeline.autoregressive_texture_reprojection import AutoregressiveTextureReprojection


class ReprojectionDataset(Dataset):
    def __init__(self, windows: Path, flow_targets: Path, ar: np.ndarray, names: list[str], indices: list[int]):
        self.windows, self.flow_targets, self.ar, self.names, self.indices = windows, flow_targets, ar, names, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            values = (
                self.ar[index],
                window["context_frames"].copy(),
                window["history_actions"].copy(),
                window["future_actions"].copy(),
                window["target_frames"].copy(),
            )
        flow = np.load(self.flow_targets / f"{Path(name).stem}.npy", allow_pickle=False).astype(np.float32, copy=False)
        if flow.shape != (8, 5, 2, 128, 128):
            raise ValueError(f"invalid all-context flow target for {name}: {flow.shape}")
        return (*values, flow.copy())


def load_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        return cache["prediction"], [str(value) for value in cache["windows"]]


def episode(name: str) -> int:
    match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
    if match is None:
        raise ValueError(f"invalid window name: {name}")
    return int(match.group(1))


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def blur(value: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    shape = value.shape[:2]
    return functional.avg_pool2d(value.flatten(0, 1), kernel, stride=1, padding=kernel // 2, count_include_pad=False).unflatten(0, shape)


def image_loss(prediction, target, context, ar, gate, residual) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    previous_target = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
    previous_prediction = torch.cat((context[:, -1:], prediction[:, :-1]), dim=1)
    target_motion = (target - previous_target).abs().mean(dim=2, keepdim=True)
    motion_weight = 1.0 + 2.0 * (target_motion >= 0.03).to(target.dtype)
    horizon = (torch.arange(1, 9, device=target.device, dtype=target.dtype) / 4.5).view(1, 8, 1, 1, 1)
    error = prediction - target
    pixel = (horizon * motion_weight * error.abs()).mean() + 0.05 * (horizon * motion_weight * error.square()).mean()
    prediction_high, target_high = prediction - blur(prediction), target - blur(target)
    texture = (horizon * motion_weight * (prediction_high - target_high).abs()).mean()
    edge_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    edge = 0.5 * ((horizon * motion_weight[..., 1:] * edge_x).mean() + (horizon * motion_weight[..., 1:, :] * edge_y).mean())
    dark_target = torch.sigmoid((0.38 - target.mean(dim=2, keepdim=True)) * 24.0)
    dark_prediction = torch.sigmoid((0.38 - prediction.mean(dim=2, keepdim=True)) * 24.0)
    dark = (horizon * (dark_prediction - dark_target).abs()).mean()
    temporal = (horizon * (((prediction - previous_prediction) - (target - previous_target)).abs())).mean()
    coarse = functional.l1_loss(functional.avg_pool2d(prediction.flatten(0, 1), 2), functional.avg_pool2d(target.flatten(0, 1), 2))
    preservation = ((1.0 - gate) * (prediction - ar).abs()).mean()
    total = pixel + 0.08 * coarse + 0.16 * edge + 0.06 * temporal + 0.18 * texture + 0.12 * dark
    total = total + 0.01 * residual.abs().mean() + 0.01 * gate.mean() + 0.01 * preservation
    return total, {"pixel": pixel, "edge": edge, "texture": texture, "dark": dark, "temporal": temporal}


def teacher_loss(model, context, target, ar, predicted_flow, source_weight, gate, teacher_flow, temperature: float, visibility_margin: float, visibility_temperature: float):
    batch, steps, sources = teacher_flow.shape[:3]
    height, width = teacher_flow.shape[-2:]
    predicted_small = functional.interpolate(predicted_flow.flatten(0, 2), (height, width), mode="bilinear", align_corners=True)
    predicted_small = predicted_small.reshape(batch, steps, sources, 2, height, width) * (width / predicted_flow.shape[-1])
    flow = functional.smooth_l1_loss(predicted_small / 16.0, teacher_flow / 16.0)
    smooth_x = (predicted_flow[..., 1:] - predicted_flow[..., :-1]).abs().mean()
    smooth_y = (predicted_flow[..., 1:, :] - predicted_flow[..., :-1, :]).abs().mean()
    smoothness = 0.5 * (smooth_x + smooth_y)
    context_small = functional.interpolate(context.flatten(0, 1), (height, width), mode="bilinear", align_corners=True)
    context_small = context_small.reshape(batch, sources, 3, height, width)
    target_small = functional.interpolate(target.flatten(0, 1), (height, width), mode="bilinear", align_corners=True)
    target_small = target_small.reshape(batch, steps, 3, height, width)
    ar_small = functional.interpolate(ar.flatten(0, 1), (height, width), mode="bilinear", align_corners=True)
    ar_small = ar_small.reshape(batch, steps, 3, height, width)
    with torch.no_grad():
        teacher_warps = model.warp(context_small, teacher_flow)
        source_error = (teacher_warps - target_small[:, :, None]).abs().mean(dim=3)
        oracle_weight = torch.softmax(-source_error / temperature, dim=2)
        teacher_reprojection = (oracle_weight[:, :, :, None] * teacher_warps).sum(dim=2)
        reproject_error = (teacher_reprojection - target_small).abs().mean(dim=2, keepdim=True)
        ar_error = (ar_small - target_small).abs().mean(dim=2, keepdim=True)
        oracle_gate = torch.sigmoid((ar_error - reproject_error - visibility_margin) / visibility_temperature)
    predicted_weight = functional.interpolate(source_weight.flatten(0, 2), (height, width), mode="bilinear", align_corners=True)
    predicted_weight = predicted_weight.reshape(batch, steps, sources, height, width).clamp_min(1e-6)
    source = -(oracle_weight * predicted_weight.log()).sum(dim=2).mean()
    predicted_gate = functional.interpolate(gate.flatten(0, 1), (height, width), mode="bilinear", align_corners=True)
    predicted_gate = predicted_gate.unflatten(0, (batch, steps)).clamp(1e-5, 1 - 1e-5)
    # Probability-form BCE is intentionally written out in float32 because
    # torch disables functional.binary_cross_entropy inside autocast regions.
    visibility = -(
        oracle_gate.float() * predicted_gate.float().log()
        + (1.0 - oracle_gate.float()) * (1.0 - predicted_gate.float()).log()
    ).mean()
    return flow, smoothness, source, visibility, oracle_gate.mean()


@torch.inference_mode()
def evaluate(loader, model, device, mean, std) -> dict[str, dict[str, float]]:
    names = ("rgb", "high_rgb", "temporal", "laplacian", "moving_rgb", "moving_temporal", "moving_laplacian", "motion_magnitude", "edge", "dark")
    sums = {key: {name: 0.0 for name in names} for key in ("autoregressive", "reprojection")}
    counts = {name: 0.0 for name in ("rgb", "high_rgb", "temporal", "moving", "motion_magnitude", "edge", "dark")}
    gate_sum = residual_sum = gate_count = 0.0
    model.eval()
    for ar, context, history, future, target, _ in loader:
        ar, context, target = (frames(value, device) for value in (ar, context, target))
        actions = torch.cat((history, future), dim=1).to(device, non_blocking=True).float()
        actions = (actions - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, _, _, gate, residual, _ = model(context, ar, actions, return_components=True)
        prediction = prediction.float().clamp(0, 1)
        previous_target = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
        target_delta = target - previous_target
        moving = target_delta.abs().mean(dim=2, keepdim=True) >= 0.03
        moving_channels = moving.expand(-1, -1, 3, -1, -1)
        high = target_delta.abs().mean(dim=(1, 2, 3, 4)) >= 0.04
        dark = target.mean(dim=2, keepdim=True) < 0.30
        dark_channels = dark.expand(-1, -1, 3, -1, -1)
        for key, value in (("autoregressive", ar), ("reprojection", prediction)):
            previous = torch.cat((context[:, -1:], value[:, :-1]), dim=1)
            error = (value - target).abs()
            delta_error = ((value - previous) - target_delta).abs()
            laplacian = ((value - blur(value)) - (target - blur(target))).abs()
            edge_x = ((value[..., 1:] - value[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
            edge_y = ((value[..., 1:, :] - value[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
            sums[key]["rgb"] += float(error.sum())
            sums[key]["high_rgb"] += float(error[high].sum())
            sums[key]["temporal"] += float(delta_error.sum())
            sums[key]["laplacian"] += float(laplacian.sum())
            sums[key]["moving_rgb"] += float(error[moving_channels].sum())
            sums[key]["moving_temporal"] += float(delta_error[moving_channels].sum())
            sums[key]["moving_laplacian"] += float(laplacian[moving_channels].sum())
            sums[key]["motion_magnitude"] += float(((value - previous).abs().mean(dim=2) - target_delta.abs().mean(dim=2)).abs().sum())
            sums[key]["edge"] += float(edge_x.sum() + edge_y.sum())
            sums[key]["dark"] += float(error[dark_channels].sum())
        counts["rgb"] += target.numel()
        counts["high_rgb"] += int(high.sum()) * int(np.prod(target.shape[1:]))
        counts["temporal"] += target.numel()
        counts["moving"] += int(moving.sum()) * 3
        counts["motion_magnitude"] += int(np.prod(target.shape[:2] + target.shape[-2:]))
        counts["edge"] += int(np.prod(target.shape[:-1])) * (target.shape[-1] - 1) + int(np.prod(target.shape[:-2])) * (target.shape[-2] - 1) * target.shape[-1]
        counts["dark"] += int(dark.sum()) * 3
        gate_sum += float(gate.float().sum())
        residual_sum += float(residual.float().abs().sum())
        gate_count += gate.numel()
    result = {}
    for key in sums:
        result[key] = {
            "rgb_mae": sums[key]["rgb"] / counts["rgb"],
            "high_motion_rgb_mae": sums[key]["high_rgb"] / max(counts["high_rgb"], 1),
            "temporal_delta_mae": sums[key]["temporal"] / counts["temporal"],
            "laplacian_mae": sums[key]["laplacian"] / counts["rgb"],
            "moving_region_rgb_mae": sums[key]["moving_rgb"] / max(counts["moving"], 1),
            "moving_region_temporal_delta_mae": sums[key]["moving_temporal"] / max(counts["moving"], 1),
            "moving_region_laplacian_mae": sums[key]["moving_laplacian"] / max(counts["moving"], 1),
            "motion_magnitude_mae": sums[key]["motion_magnitude"] / counts["motion_magnitude"],
            "edge_mae": sums[key]["edge"] / counts["edge"],
            "dark_region_rgb_mae": sums[key]["dark"] / max(counts["dark"], 1),
        }
    result["reprojection"]["gate_mean"] = gate_sum / gate_count
    result["reprojection"]["residual_abs_mean"] = residual_sum / (gate_count * 3)
    return result


def score(result: dict[str, dict[str, float]]) -> float:
    baseline, candidate = result["autoregressive"], result["reprojection"]
    weights = {"rgb_mae": 0.25, "high_motion_rgb_mae": 0.10, "temporal_delta_mae": 0.08, "laplacian_mae": 0.15, "moving_region_rgb_mae": 0.10, "moving_region_temporal_delta_mae": 0.07, "moving_region_laplacian_mae": 0.12, "motion_magnitude_mae": 0.03, "edge_mae": 0.05, "dark_region_rgb_mae": 0.05}
    value = sum(weight * candidate[name] / baseline[name] for name, weight in weights.items())
    value += 10.0 * max(candidate["rgb_mae"] / baseline["rgb_mae"] - 1.002, 0.0)
    value += 10.0 * max(candidate["high_motion_rgb_mae"] / baseline["high_motion_rgb_mae"] - 1.002, 0.0)
    return float(value)


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(output: Path, model, manifest: dict, normalization: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_save({"format": "track2-autoregressive-texture-reprojection-v1", "state_dict": model.state_dict()}, output / "model.pt")
    with (output / "autoregressive_texture_reprojection_config.npz.tmp").open("wb") as handle:
        np.savez(handle, base_channels=model.base_channels, residual_scale=model.residual_scale, context_frames=5, prediction_frames=8, action_dim=14)
    os.replace(output / "autoregressive_texture_reprojection_config.npz.tmp", output / "autoregressive_texture_reprojection_config.npz")
    shutil.copy2(normalization, output / "action_normalization.npz")
    temporary = output / f"training_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "training_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--autoregressive-cache", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--action-normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--dev-episode-count", type=int, default=4)
    parser.add_argument("--flow-loss-weight", type=float, default=0.5)
    parser.add_argument("--source-loss-weight", type=float, default=0.08)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.05)
    parser.add_argument("--teacher-temperature", type=float, default=0.025)
    parser.add_argument("--visibility-margin", type=float, default=0.01)
    parser.add_argument("--visibility-temperature", type=float, default=0.005)
    parser.add_argument("--initial-checkpoint", help="Optional compatible reprojection checkpoint used to initialize a new run.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ar, names = load_cache(Path(args.autoregressive_cache))
    split = json.loads(Path(args.split_manifest).read_text())
    allowed = set(split["train_episodes"])
    if any(episode(name) not in allowed for name in names):
        raise ValueError("autoregressive cache contains non-train windows")
    rng = np.random.default_rng(args.seed)
    dev_episodes = sorted(rng.choice(split["train_episodes"], args.dev_episode_count, replace=False).tolist())
    train_indices = [index for index, name in enumerate(names) if episode(name) not in dev_episodes]
    dev_indices = [index for index, name in enumerate(names) if episode(name) in dev_episodes]
    train = ReprojectionDataset(Path(args.windows), Path(args.flow_targets), ar, names, train_indices)
    dev = ReprojectionDataset(Path(args.windows), Path(args.flow_targets), ar, names, dev_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=2, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    normalization_path = Path(args.action_normalization)
    normalization = np.load(normalization_path, allow_pickle=False)
    device = torch.device(args.device)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)
    torch.manual_seed(args.seed)
    model = AutoregressiveTextureReprojection(args.base_channels).to(device)
    if args.initial_checkpoint and not args.resume:
        initial_state = torch.load(Path(args.initial_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
        if initial_state.get("format") != "track2-autoregressive-texture-reprojection-v1":
            raise ValueError("unsupported initial reprojection checkpoint")
        model.load_state_dict(initial_state["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    config = {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "validation_interval": args.validation_interval, "seed": args.seed, "dev_episodes": dev_episodes, "train_sample_count": len(train), "dev_sample_count": len(dev), "flow_loss_weight": args.flow_loss_weight, "source_loss_weight": args.source_loss_weight, "visibility_loss_weight": args.visibility_loss_weight, "teacher_temperature": args.teacher_temperature, "visibility_margin": args.visibility_margin, "visibility_temperature": args.visibility_temperature, "initial_checkpoint": str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None, "base_channels": args.base_channels}
    state_path = output / "training_state.pt"
    start, best, history = 0, float("inf"), []
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("format") != "track2-autoregressive-texture-reprojection-training-v1" or state.get("config") != config:
            raise ValueError("reprojection resume state does not match")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = int(state["step"]), float(state["best_score"]), list(state["history"])
    baseline = evaluate(dev_loader, model, device, mean, std)
    if not history:
        best = score(baseline)
        history.append({"step": 0, "selection_score": best, **baseline})
        save_checkpoint(output / "best", model, {"format": "track2-autoregressive-texture-reprojection-v1", **config, "checkpoint_step": 0, "best_selection_score": best, "validation": history, "autoregressive_cache": str(Path(args.autoregressive_cache).resolve()), "flow_targets": str(Path(args.flow_targets).resolve())}, normalization_path)
    iterator = iter(train_loader)
    model.train()
    for step in range(start + 1, args.steps + 1):
        try:
            ar_batch, context, history_actions, future, target, teacher_flow = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            ar_batch, context, history_actions, future, target, teacher_flow = next(iterator)
        ar_batch, context, target = (frames(value, device) for value in (ar_batch, context, target))
        actions = torch.cat((history_actions, future), dim=1).to(device, non_blocking=True).float()
        actions = (actions - mean) / std
        teacher_flow = teacher_flow.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, flow, source_weight, gate, residual, _ = model(context, ar_batch, actions, return_components=True)
            prediction = prediction.clamp(0, 1)
            reconstruction, components = image_loss(prediction, target, context, ar_batch, gate, residual)
            flow_loss, smoothness, source_loss, visibility_loss, oracle_gate = teacher_loss(model, context.float(), target.float(), ar_batch.float(), flow.float(), source_weight.float(), gate.float(), teacher_flow.float(), args.teacher_temperature, args.visibility_margin, args.visibility_temperature)
            loss = reconstruction + args.flow_loss_weight * flow_loss + 0.001 * smoothness + args.source_loss_weight * source_loss + args.visibility_loss_weight * visibility_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "reconstruction": float(reconstruction.detach()), "flow": float(flow_loss.detach()), "source": float(source_loss.detach()), "visibility": float(visibility_loss.detach()), "oracle_gate": float(oracle_gate.detach()), "gate": float(gate.detach().mean()), **{name: float(value.detach()) for name, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(dev_loader, model, device, mean, std)
            metric = score(result)
            record = {"step": step, "selection_score": metric, **result}
            history.append(record)
            manifest = {"format": "track2-autoregressive-texture-reprojection-v1", **config, "checkpoint_step": step, "best_selection_score": min(best, metric), "validation": history, "autoregressive_cache": str(Path(args.autoregressive_cache).resolve()), "flow_targets": str(Path(args.flow_targets).resolve())}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, manifest, normalization_path)
            if metric < best:
                best = metric
                manifest["best_checkpoint_step"] = step
                save_checkpoint(output / "best", model, manifest, normalization_path)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_interval == 0 or step == args.steps:
            atomic_save({"format": "track2-autoregressive-texture-reprojection-training-v1", "config": config, "step": step, "best_score": best, "history": history, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, state_path)


if __name__ == "__main__":
    main()
