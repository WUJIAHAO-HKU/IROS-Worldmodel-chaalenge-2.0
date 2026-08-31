#!/usr/bin/env python3
"""Train a pure transport model with motion-balanced RAFT supervision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wam_pipeline.motion_weighted_flow_renderer import MotionWeightedFlowRenderer


class Windows(Dataset):
    def __init__(self, root: Path, names: list[str], flows: Path | None = None):
        self.root, self.names, self.flows = root, names, flows

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with np.load(self.root / name, allow_pickle=False) as value:
            result = [value["context_frames"].copy(), value["history_actions"].copy(), value["future_actions"].copy(), value["target_frames"].copy()]
        if self.flows is not None:
            result.append(np.load(self.flows / f"{Path(name).stem}.npy").astype(np.float32))
        return tuple(result)


def frames(value, device):
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255)


def losses(prediction, flow, target, teacher, context_last):
    magnitude = teacher.square().sum(dim=2, keepdim=True).add(1e-4).sqrt()
    motion_weight = 1.0 + 15.0 * (magnitude / 2.0).clamp(0, 1)
    endpoint = (flow - teacher).square().sum(dim=2, keepdim=True).add(0.01).sqrt()
    flow_loss = (motion_weight * endpoint).sum() / motion_weight.sum()
    target_128 = functional.interpolate(target.flatten(0, 1), (128, 128), mode="area").unflatten(0, target.shape[:2])
    last_128 = functional.interpolate(context_last, (128, 128), mode="area")
    warped_128 = MotionWeightedFlowRenderer.warp(last_128, flow)
    changed = (target_128 - last_128[:, None]).abs().mean(dim=2, keepdim=True)
    rgb_weight = 1.0 + 7.0 * (changed / 0.05).clamp(0, 1)
    rgb = (rgb_weight * (warped_128 - target_128).abs()).sum() / (rgb_weight.sum() * 3)
    edge = functional.l1_loss(
        warped_128[..., 1:] - warped_128[..., :-1], target_128[..., 1:] - target_128[..., :-1]
    ) + functional.l1_loss(
        warped_128[..., 1:, :] - warped_128[..., :-1, :], target_128[..., 1:, :] - target_128[..., :-1, :]
    )
    smooth = (flow[..., 1:] - flow[..., :-1]).abs().mean() + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    return flow_loss + 3.0 * rgb + 0.15 * edge + 0.002 * smooth, flow_loss, rgb


def evaluate(model, loader, device, mean, std):
    totals = {"rgb": 0.0, "highpass": 0.0, "dark": 0.0}
    count = dark_count = 0
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            context, target = frames(context, device), frames(target, device)
            actions = (torch.cat((history, future), 1).to(device).float() - mean) / std
            prediction, _ = model(context, actions)
            error = (prediction.float().clamp(0, 1) - target).abs()
            totals["rgb"] += float(error.sum()); count += error.numel()
            hp = lambda x: x.flatten(0, 1) - functional.avg_pool2d(x.flatten(0, 1), 5, 1, 2)
            totals["highpass"] += float((hp(prediction.float()) - hp(target)).abs().sum())
            dark = target.mean(2, keepdim=True) < 0.32
            totals["dark"] += float((error * dark).sum()); dark_count += int(dark.sum()) * 3
    return {"rgb_mae_0_255": 255 * totals["rgb"] / count, "highpass_mae_0_255": 255 * totals["highpass"] / count, "dark_rgb_mae_0_255": 255 * totals["dark"] / dark_count}


def save(root, model, mean, std, manifest):
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-motion-weighted-flow-renderer-v1", "state_dict": model.state_dict()}, root / "model.pt")
    np.savez(root / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(root / "motion_weighted_flow_config.npz", base_channels=model.base_channels, max_flow_128=model.max_flow_128)
    (root / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--flow-targets", required=True); parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-samples", type=int, default=64); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(20260807); np.random.seed(20260807)
    root, split = Path(args.windows), json.loads(Path(args.split_manifest).read_text())
    names = lambda episodes: [p.name for e in episodes for p in sorted(root.glob(f"episode{e}_*.npz"))]
    train_names, val_names = names(split["train_episodes"]), names(split["validation_episodes"])
    val_names = [val_names[i] for i in np.linspace(0, len(val_names)-1, min(args.validation_samples, len(val_names)), dtype=int)]
    flow_root = Path(args.flow_targets)
    magnitudes = np.asarray([float(np.abs(np.load(flow_root / f"{Path(n).stem}.npy", mmap_mode="r")).mean()) for n in train_names])
    weights = 1.0 + 3.0 * (magnitudes >= np.quantile(magnitudes, 0.75))
    sampler = WeightedRandomSampler(weights, len(train_names), replacement=True, generator=torch.Generator().manual_seed(20260807))
    train_loader = DataLoader(Windows(root, train_names, flow_root), batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Windows(root, val_names), batch_size=1, num_workers=1)
    normalization = np.load(args.normalization); device = torch.device(args.device)
    mean = torch.from_numpy(normalization["mean"].astype(np.float32)).to(device); std = torch.from_numpy(normalization["std"].astype(np.float32)).to(device)
    model = MotionWeightedFlowRenderer(args.base_channels).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); best = float("inf"); iterator = iter(train_loader); history = []
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        context, history_actions, future_actions, target, teacher = batch
        context, target, teacher = frames(context, device), frames(target, device), teacher.to(device).float()
        actions = (torch.cat((history_actions, future_actions), 1).to(device).float() - mean) / std
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, flow = model(context, actions); loss, flow_loss, rgb = losses(prediction, flow.float(), target, teacher, context[:, -1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        if step % 25 == 0: print(json.dumps({"step":step,"loss":float(loss),"flow":float(flow_loss),"rgb":float(rgb),"gpu_mb":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = {"step": step, **evaluate(model, val_loader, device, mean, std)}; history.append(metrics); print(json.dumps({"validation":metrics}), flush=True)
            if metrics["rgb_mae_0_255"] < best:
                best = metrics["rgb_mae_0_255"]; save(output / "best", model, mean, std, {"format":"track2-motion-weighted-flow-renderer-v1","config":vars(args),"validation":history,"best_rgb":best})
            model.train()
    save(output, model, mean, std, {"format":"track2-motion-weighted-flow-renderer-v1","config":vars(args),"validation":history,"best_rgb":best})


if __name__ == "__main__": main()
