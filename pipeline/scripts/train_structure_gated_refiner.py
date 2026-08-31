#!/usr/bin/env python3
"""Train a zero-initialized structure-mask-gated high-frequency repair branch."""

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

from wam_pipeline.structure_gated_refiner import StructureGatedHighFrequencyRefiner


def load_prediction_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        return cache["prediction"], [str(value) for value in cache["windows"]]


def load_mask_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        return cache["structure_mask"], [str(value) for value in cache["windows"]]


def episode(name: str) -> int:
    match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
    if match is None:
        raise ValueError(f"invalid window name: {name}")
    return int(match.group(1))


class RepairDataset(Dataset):
    def __init__(self, windows: Path, baseline, structure, masks, names, indices):
        self.windows, self.baseline, self.structure = windows, baseline, structure
        self.masks, self.names, self.indices = masks, names, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        with np.load(self.windows / self.names[index], allow_pickle=False) as window:
            context = window["context_frames"][-1:].copy()
            actions = window["future_actions"].copy()
            target = window["target_frames"].copy()
        return self.baseline[index], self.structure[index], self.masks[index], context, actions, target


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    blurred = functional.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
    return (flat - blurred).unflatten(0, value.shape[:2])


def repair_loss(prediction, target, context, baseline, mask, correction):
    previous_target = torch.cat((context, target[:, :-1]), dim=1)
    previous_prediction = torch.cat((context, prediction[:, :-1]), dim=1)
    target_motion = (target - previous_target).abs().mean(dim=2, keepdim=True)
    moving = (target_motion >= 0.03).to(target.dtype)
    horizon = torch.arange(1, 9, device=target.device, dtype=target.dtype)
    horizon = (horizon / horizon.mean()).view(1, 8, 1, 1, 1)
    motion_weight = 1.0 + 1.5 * moving
    structure_weight = 1.0 + 3.0 * mask
    error = prediction - target
    pixel = (horizon * motion_weight * error.abs()).mean() + 0.05 * (horizon * motion_weight * error.square()).mean()
    texture_error = (highpass(prediction) - highpass(target)).abs()
    texture = (horizon * motion_weight * structure_weight * texture_error).mean()
    edge_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    edge = 0.5 * (
        (horizon * motion_weight[..., 1:] * structure_weight[..., 1:] * edge_x).mean()
        + (horizon * motion_weight[..., 1:, :] * structure_weight[..., 1:, :] * edge_y).mean()
    )
    temporal = ((prediction - previous_prediction) - (target - previous_target)).abs().mean()
    target_dark = torch.sigmoid((0.35 - target.mean(dim=2, keepdim=True)) * 24.0)
    dark = (horizon * mask * target_dark * error.abs()).sum() / (horizon.mul(mask).mul(target_dark).sum() * 3.0).clamp_min(1.0)
    outside = 1.0 - mask
    preservation = (outside * (prediction - baseline).abs()).sum() / (outside.sum() * 3.0).clamp_min(1.0)
    correction_penalty = correction.abs().mean()
    total = pixel + 0.12 * temporal + 0.25 * texture + 0.22 * edge + 0.10 * dark + 0.10 * preservation + 0.005 * correction_penalty
    return total, {"pixel": pixel, "temporal": temporal, "texture": texture, "edge": edge, "dark": dark, "preservation": preservation, "correction_abs": correction_penalty}


@torch.inference_mode()
def evaluate(loader, model, device, mean, std):
    metric_names = ("rgb", "high_rgb", "temporal", "highpass", "edge", "dark", "moving_rgb", "moving_temporal", "moving_laplacian")
    sums = {model_name: {name: 0.0 for name in metric_names} for model_name in ("baseline", "refiner")}
    counts = {name: 0.0 for name in ("rgb", "high_rgb", "temporal", "highpass", "edge", "dark", "moving")}
    gate_sum = correction_sum = support_sum = gate_count = 0.0
    model.eval()
    for baseline, structure, mask, context, actions, target in loader:
        baseline, structure, context, target = (frames(value, device) for value in (baseline, structure, context, target))
        mask = mask[:, :, None].to(device, non_blocking=True).float().div(255.0)
        actions = (actions.to(device, non_blocking=True).float() - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, gate, correction, support = model(context, baseline, structure, mask, actions)
        previous_target = torch.cat((context, target[:, :-1]), dim=1)
        target_delta = target - previous_target
        moving = target_delta.abs().mean(dim=2, keepdim=True) >= 0.03
        moving_channels = moving.expand(-1, -1, 3, -1, -1)
        high = target_delta.abs().mean(dim=(1, 2, 3, 4)) >= 0.04
        dark = (target.mean(dim=2, keepdim=True) < 0.30).expand_as(target)
        for name, value in (("baseline", baseline), ("refiner", prediction.float())):
            previous = torch.cat((context, value[:, :-1]), dim=1)
            error = (value - target).abs()
            temporal = ((value - previous) - target_delta).abs()
            hp = (highpass(value) - highpass(target)).abs()
            edge_x = ((value[..., 1:] - value[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
            edge_y = ((value[..., 1:, :] - value[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
            sums[name]["rgb"] += float(error.sum())
            sums[name]["high_rgb"] += float(error[high].sum())
            sums[name]["temporal"] += float(temporal.sum())
            sums[name]["highpass"] += float(hp.sum())
            sums[name]["edge"] += float(edge_x.sum() + edge_y.sum())
            sums[name]["dark"] += float(error[dark].sum())
            sums[name]["moving_rgb"] += float(error[moving_channels].sum())
            sums[name]["moving_temporal"] += float(temporal[moving_channels].sum())
            sums[name]["moving_laplacian"] += float(hp[moving_channels].sum())
        counts["rgb"] += target.numel()
        counts["high_rgb"] += int(high.sum()) * int(np.prod(target.shape[1:]))
        counts["temporal"] += target.numel()
        counts["highpass"] += target.numel()
        counts["edge"] += edge_x.numel() + edge_y.numel()
        counts["dark"] += int(dark.sum())
        counts["moving"] += int(moving.sum()) * 3
        gate_sum += float(gate.float().sum())
        correction_sum += float(correction.float().abs().sum())
        support_sum += float(support.float().sum())
        gate_count += gate.numel()
    result = {}
    mapping = {"rgb_mae": ("rgb", "rgb"), "high_motion_rgb_mae": ("high_rgb", "high_rgb"), "temporal_delta_mae": ("temporal", "temporal"), "highpass_mae": ("highpass", "highpass"), "edge_mae": ("edge", "edge"), "dark_region_rgb_mae": ("dark", "dark"), "moving_region_rgb_mae": ("moving_rgb", "moving"), "moving_region_temporal_delta_mae": ("moving_temporal", "moving"), "moving_region_laplacian_mae": ("moving_laplacian", "moving")}
    for model_name in sums:
        result[model_name] = {out: sums[model_name][source] / max(counts[count], 1.0) for out, (source, count) in mapping.items()}
    result["refiner"].update({"gate_mean": gate_sum / gate_count, "support_mean": support_sum / gate_count, "correction_abs_mean": correction_sum / (gate_count * 3.0)})
    return result


def selection_metric(result):
    baseline, candidate = result["baseline"], result["refiner"]
    weights = {"rgb_mae": 0.30, "high_motion_rgb_mae": 0.15, "temporal_delta_mae": 0.10, "highpass_mae": 0.15, "edge_mae": 0.10, "dark_region_rgb_mae": 0.05, "moving_region_rgb_mae": 0.05, "moving_region_temporal_delta_mae": 0.05, "moving_region_laplacian_mae": 0.05}
    score = sum(weight * candidate[name] / baseline[name] for name, weight in weights.items())
    for name in ("rgb_mae", "high_motion_rgb_mae"):
        score += 20.0 * max(candidate[name] / baseline[name] - 1.0002, 0.0)
    return float(score)


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(output, model, normalization_path, manifest):
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"format": "track2-structure-gated-high-frequency-refiner-v1", "state_dict": model.state_dict()}, output / "model.pt")
    with (output / "structure_refiner_config.npz.tmp").open("wb") as handle:
        np.savez(handle, base_channels=model.base_channels, residual_scale=model.residual_scale, prediction_frames=8, action_dim=14)
    os.replace(output / "structure_refiner_config.npz.tmp", output / "structure_refiner_config.npz")
    shutil.copy2(normalization_path, output / "action_normalization.npz")
    temporary = output / f"training_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "training_manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--mask-cache", required=True)
    parser.add_argument("--action-normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--dev-episode-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    baseline, names = load_prediction_cache(Path(args.baseline_cache))
    structure, structure_names = load_prediction_cache(Path(args.structure_cache))
    masks, mask_names = load_mask_cache(Path(args.mask_cache))
    if names != structure_names or names != mask_names or baseline.shape != structure.shape or masks.shape != baseline.shape[:2] + baseline.shape[2:4]:
        raise ValueError("baseline, structure, and mask caches are not aligned")
    split = json.loads(Path(args.split_manifest).read_text())
    rng = np.random.default_rng(args.seed)
    dev_episodes = sorted(rng.choice(split["train_episodes"], args.dev_episode_count, replace=False).tolist())
    train_indices = [index for index, name in enumerate(names) if episode(name) not in dev_episodes]
    dev_indices = [index for index, name in enumerate(names) if episode(name) in dev_episodes]
    train = RepairDataset(Path(args.windows), baseline, structure, masks, names, train_indices)
    dev = RepairDataset(Path(args.windows), baseline, structure, masks, names, dev_indices)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=2, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    normalization_path = Path(args.action_normalization)
    normalization = np.load(normalization_path, allow_pickle=False)
    mean = torch.from_numpy(np.asarray(normalization["mean"], np.float32)).to(device)
    std = torch.from_numpy(np.asarray(normalization["std"], np.float32)).to(device)
    torch.manual_seed(args.seed)
    model = StructureGatedHighFrequencyRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    config = {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "validation_interval": args.validation_interval, "seed": args.seed, "dev_episodes": dev_episodes, "train_sample_count": len(train_indices), "dev_sample_count": len(dev_indices), "baseline_cache": str(Path(args.baseline_cache).resolve()), "structure_cache": str(Path(args.structure_cache).resolve()), "mask_cache": str(Path(args.mask_cache).resolve())}
    initial = evaluate(dev_loader, model, device, mean, std)
    history = [{"step": 0, "selection_metric": 1.0, **initial}]
    best = 1.0
    save_checkpoint(output / "best", model, normalization_path, {"format": "track2-structure-gated-high-frequency-refiner-v1", **config, "checkpoint_step": 0, "best_selection_metric": best, "validation": history})
    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            baseline_batch, structure_batch, mask, context, actions, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            baseline_batch, structure_batch, mask, context, actions, target = next(iterator)
        baseline_batch, structure_batch, context, target = (frames(value, device) for value in (baseline_batch, structure_batch, context, target))
        mask = mask[:, :, None].to(device, non_blocking=True).float().div(255.0)
        actions = (actions.to(device, non_blocking=True).float() - mean) / std
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, gate, correction, support = model(context, baseline_batch, structure_batch, mask, actions)
            loss, components = repair_loss(prediction, target, context, baseline_batch, mask, correction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "gate_mean": float(gate.detach().mean()), "support_mean": float(support.detach().mean()), **{name: float(value.detach()) for name, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(dev_loader, model, device, mean, std)
            metric = selection_metric(result)
            record = {"step": step, "selection_metric": metric, **result}
            history.append(record)
            manifest = {"format": "track2-structure-gated-high-frequency-refiner-v1", **config, "checkpoint_step": step, "best_selection_metric": min(best, metric), "validation": history}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, normalization_path, manifest)
            if metric < best:
                best = metric
                manifest["best_checkpoint_step"] = step
                save_checkpoint(output / "best", model, normalization_path, manifest)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_interval == 0 or step == args.steps:
            atomic_torch_save({"format": "track2-structure-refiner-training-state-v1", "config": config, "step": step, "best_metric": best, "history": history, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, output / "training_state.pt")


if __name__ == "__main__":
    main()
