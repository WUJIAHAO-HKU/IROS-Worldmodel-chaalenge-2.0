#!/usr/bin/env python3
"""Train a v8-anchored router over deployable predicted texture transports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.hybrid_transport_router import HybridTransportRouter
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet


def episode(name: str) -> int:
    match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
    if match is None:
        raise ValueError(name)
    return int(match.group(1))


class DatasetV9(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int]) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        with np.load(self.windows / self.names[index], allow_pickle=False) as window:
            return (
                self.parent[index], window["context_frames"].copy(), window["history_actions"].copy(),
                window["future_actions"].copy(), window["target_frames"].copy(), self.names[index],
            )


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


@torch.no_grad()
def candidates_from_flow(model, parent, context, actions, mean, std):
    normalized = (actions - mean) / std
    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16, enabled=context.device.type == "cuda"):
        fused, flow, _ = model(context, normalized, return_flow=True)
        warped = model._warp(context, flow)
    return torch.cat((parent[:, :, None], warped.float().clamp(0, 1), fused[:, :, None].float().clamp(0, 1)), dim=2)


def active_statistics(dataset: DatasetV9) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for item in range(len(dataset)):
        _, _, history, future, _, _ = dataset[item]
        actions = torch.from_numpy(np.concatenate((history, future), axis=0))[None]
        active, _ = HybridTransportRouter.active_arm_actions(actions)
        values.append(active.squeeze(0))
    value = torch.cat(values)
    return value.mean(0), value.std(0).clamp_min(1e-5)


def loss_function(prediction, logits, weights, candidates, context, target):
    b, t, k = candidates.shape[:3]
    block_error = (candidates - target[:, :, None]).abs().mean(dim=3)
    block_error = F.avg_pool2d(block_error.flatten(0, 1), 4, stride=4).unflatten(0, (b, t))
    best_other_error, best_other = block_error[:, :, 1:].min(dim=2)
    # Keep v8 unless another source wins by at least 0.5 RGB level.
    target_choice = torch.where(best_other_error + 0.5 / 255.0 < block_error[:, :, 0], best_other + 1, 0)
    classification = F.cross_entropy(logits.flatten(0, 1), target_choice.flatten(0, 1), reduction="none").unflatten(0, (b, t))
    moving = (target - torch.cat((context[:, -1:], target[:, :-1]), dim=1)).abs().mean(dim=2)
    moving = F.avg_pool2d(moving.flatten(0, 1)[:, None], 4, stride=4).unflatten(0, (b, t)).squeeze(2)
    classification = (classification * (1 + 2 * (moving >= 0.03))).mean()
    # Regress each transported candidate's measurable advantage over v8.  This
    # retains signal even when v8 wins most blocks, where winner-only CE tends
    # to collapse to the parent everywhere.
    target_advantage = ((block_error[:, :, :1] - block_error) * 25.5).clamp(-5.0, 5.0)
    predicted_advantage = logits - logits[:, :, :1]
    advantage_error = F.smooth_l1_loss(
        predicted_advantage[:, :, 1:], target_advantage[:, :, 1:], beta=0.25, reduction="none"
    )
    advantage_weight = 1.0 + 3.0 * (target_advantage[:, :, 1:] > 0).to(advantage_error.dtype)
    advantage_weight = advantage_weight * (1.0 + moving[:, :, None])
    advantage = (advantage_error * advantage_weight).sum() / advantage_weight.sum().clamp_min(1.0)
    expected = (weights * block_error).sum(dim=2).mean()
    pixel = (prediction - target).abs().mean()
    parent = candidates[:, :, 0]
    target_high = target.flatten(0, 1) - F.avg_pool2d(target.flatten(0, 1), 5, 1, 2)
    pred_high = prediction.flatten(0, 1) - F.avg_pool2d(prediction.flatten(0, 1), 5, 1, 2)
    texture = (pred_high - target_high).abs().mean()
    change = (prediction - parent).abs().mean()
    total = pixel + 0.25 * expected + 0.02 * classification + 0.08 * advantage + 0.10 * texture + 0.002 * change
    return total, {"pixel": pixel, "expected": expected, "classification": classification, "advantage": advantage,
                   "texture": texture, "change": change}


@torch.inference_mode()
def evaluate(loader, router, flow_model, device, flow_mean, flow_std, active_mean, active_std):
    margins = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    sums = {margin: 0.0 for margin in margins}
    baseline_sum = oracle_sum = 0.0
    count = 0
    router.eval(); flow_model.eval()
    for parent, context, history, future, target, _ in loader:
        parent, context, target = (frames(value, device) for value in (parent, context, target))
        actions = torch.cat((history, future), dim=1).to(device).float()
        candidates = candidates_from_flow(flow_model, parent, context, actions, flow_mean, flow_std)
        active, arm = HybridTransportRouter.active_arm_actions(actions)
        active = (active - active_mean) / active_std
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, logits, _ = router(candidates, context, active, arm)
        error = (candidates - target[:, :, None]).abs().mean(dim=3)
        pooled = F.avg_pool2d(error.flatten(0, 1), 4, stride=4).unflatten(0, target.shape[:2])
        oracle_choice = pooled.argmin(dim=2).repeat_interleave(4, 2).repeat_interleave(4, 3)
        oracle = candidates.gather(2, oracle_choice[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        baseline_sum += float((parent - target).abs().sum()); oracle_sum += float((oracle - target).abs().sum())
        other_logit, other_choice = logits[:, :, 1:].max(dim=2)
        for margin in margins:
            choice = torch.where(other_logit - logits[:, :, 0] > margin, other_choice + 1, 0)
            full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            routed = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            sums[margin] += float((routed - target).abs().sum())
        count += target.numel()
    baseline = 255 * baseline_sum / count
    metrics = {"baseline_rgb_mae": baseline, "block4_oracle_rgb_mae": 255 * oracle_sum / count}
    metrics["margin_rgb_mae"] = {str(margin): 255 * sums[margin] / count for margin in margins}
    best_margin = min(margins, key=lambda value: sums[value])
    metrics.update({"best_margin": best_margin, "router_rgb_mae": metrics["margin_rgb_mae"][str(best_margin)]})
    metrics["relative_improvement_percent"] = 100 * (baseline - metrics["router_rgb_mae"]) / baseline
    return metrics


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--dev-episodes", type=int, default=2)
    parser.add_argument("--dev-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    episodes = sorted({episode(name) for name in names})
    rng = np.random.default_rng(args.seed)
    dev_episodes = set(rng.choice(episodes, min(args.dev_episodes, len(episodes) - 1), replace=False).tolist())
    train_indices = [index for index, name in enumerate(names) if episode(name) not in dev_episodes]
    all_dev = [index for index, name in enumerate(names) if episode(name) in dev_episodes]
    dev_indices = [all_dev[index] for index in np.linspace(0, len(all_dev) - 1, min(args.dev_samples, len(all_dev)), dtype=int)]
    train = DatasetV9(Path(args.windows), parent, names, train_indices)
    dev = DatasetV9(Path(args.windows), parent, names, dev_indices)
    active_mean, active_std = active_statistics(train)
    device = torch.device(args.device)
    flow_checkpoint = Path(args.flow_checkpoint)
    with np.load(flow_checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        flow_model = MultiSourceActionFlowUNet(int(config["base_channels"]))
    state = torch.load(flow_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    flow_model.load_state_dict(state["state_dict"], strict=True)
    flow_model = flow_model.to(device).eval().requires_grad_(False)
    with np.load(flow_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        flow_mean = torch.from_numpy(normalization["mean"]).to(device)
        flow_std = torch.from_numpy(normalization["std"]).to(device)
    active_mean, active_std = active_mean.to(device), active_std.to(device)
    router = HybridTransportRouter().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    common = {"batch_size": args.batch_size, "num_workers": 2, "pin_memory": True}
    train_loader = DataLoader(train, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), **common)
    dev_loader = DataLoader(dev, shuffle=False, **common)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []
    initial = evaluate(dev_loader, router, flow_model, device, flow_mean, flow_std, active_mean, active_std)
    history.append({"step": 0, **initial}); best = initial["router_rgb_mae"]
    atomic_save({"format": "track2-hybrid-transport-router-v9", "state_dict": router.state_dict(), "step": 0,
                 "metrics": initial, "active_mean": active_mean.cpu(), "active_std": active_std.cpu()}, output / "best.pt")
    print(json.dumps(history[-1]), flush=True)
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        router.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        parent_batch, context, history_actions, future, target, _ = batch
        parent_batch, context, target = (frames(value, device) for value in (parent_batch, context, target))
        actions = torch.cat((history_actions, future), dim=1).to(device).float()
        candidates = candidates_from_flow(flow_model, parent_batch, context, actions, flow_mean, flow_std)
        active, arm = HybridTransportRouter.active_arm_actions(actions)
        active = (active - active_mean) / active_std
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, logits, weights = router(candidates, context, active, arm)
            loss, components = loss_function(prediction, logits, weights, candidates, context, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss), **{key: float(value) for key, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, router, flow_model, device, flow_mean, flow_std, active_mean, active_std)
            history.append({"step": step, **metrics}); print(json.dumps(history[-1]), flush=True)
            if metrics["router_rgb_mae"] < best:
                best = metrics["router_rgb_mae"]
                atomic_save({"format": "track2-hybrid-transport-router-v9", "state_dict": router.state_dict(), "step": step,
                             "metrics": metrics, "active_mean": active_mean.cpu(), "active_std": active_std.cpu()}, output / "best.pt")
    manifest = {"format": "track2-hybrid-transport-router-v9", "train_samples": len(train), "dev_samples": len(dev),
                "dev_episodes": sorted(dev_episodes), "flow_checkpoint": str(flow_checkpoint.resolve()), "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
