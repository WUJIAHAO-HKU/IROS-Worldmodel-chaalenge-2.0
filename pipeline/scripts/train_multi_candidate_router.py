#!/usr/bin/env python3
"""Train and evaluate a target-free spatial router over three video predictors."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from wam_pipeline.multi_candidate_router import MultiCandidateRouter


def load_cache(path: str) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        return cache["prediction"], cache["windows"].astype(str).tolist()


def load_candidates(paths: list[str]) -> tuple[np.ndarray, list[str]]:
    loaded = [load_cache(path) for path in paths]
    names = loaded[0][1]
    if any(item[1] != names or item[0].shape != loaded[0][0].shape for item in loaded[1:]):
        raise ValueError("candidate caches are not aligned")
    return np.stack([item[0] for item in loaded], axis=1), names


def episode(name: str) -> int:
    match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
    if match is None:
        raise ValueError(f"invalid window name {name}")
    return int(match.group(1))


class RouterDataset(Dataset):
    def __init__(self, windows: Path, candidates: np.ndarray, names: list[str], indices: list[int], action_mean, action_std) -> None:
        self.windows, self.candidates, self.names, self.indices = windows, candidates, names, indices
        self.action_mean, self.action_std = action_mean, action_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        with np.load(self.windows / self.names[index], allow_pickle=False) as window:
            context = window["context_frames"].copy()
            actions = ((window["future_actions"].astype(np.float32) - self.action_mean) / self.action_std).copy()
            target = window["target_frames"].copy()
        return self.candidates[index], context, actions, target


def to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=True).float().div(255.0)


def prepare(candidates, context, actions, target, device):
    # input candidates: B,K,T,H,W,C; model candidates: B,T,K,C,H,W
    candidates = to_device(candidates, device).permute(0, 2, 1, 5, 3, 4)
    context = to_device(context, device).permute(0, 1, 4, 2, 3)
    actions = actions.to(device, non_blocking=True).float()
    target = to_device(target, device).permute(0, 1, 4, 2, 3)
    return candidates, context, actions, target


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return (flat - F.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)).unflatten(0, value.shape[:2])


def router_loss(prediction, weights, candidates, context, target):
    b, t = target.shape[:2]
    per_candidate = (candidates - target[:, :, None]).abs().mean(dim=3)
    smooth_error = F.avg_pool2d(per_candidate.flatten(0, 1), 3, stride=1, padding=1, count_include_pad=False).unflatten(0, (b, t))
    soft_target = (-smooth_error / 0.0125).softmax(dim=2)
    classification = -(soft_target * weights.clamp_min(1e-7).log()).sum(dim=2).mean()
    pixel = (weights * per_candidate).sum(dim=2).mean()
    target_hp = highpass(target)
    candidate_hp = torch.stack([highpass(candidates[:, :, index]) for index in range(3)], dim=2)
    texture_error = (candidate_hp - target_hp[:, :, None]).abs().mean(dim=3)
    texture = (weights * texture_error).sum(dim=2).mean()
    previous_target = torch.cat((context[:, -1:,], target[:, :-1]), dim=1)
    target_delta = target - previous_target
    temporal_errors = []
    for index in range(3):
        previous_candidate = torch.cat((context[:, -1:], candidates[:, :-1, index]), dim=1)
        temporal_errors.append(((candidates[:, :, index] - previous_candidate) - target_delta).abs().mean(dim=2))
    temporal_error = torch.stack(temporal_errors, dim=2)
    temporal = (weights * temporal_error).sum(dim=2).mean()
    total = pixel + 0.25 * texture + 0.10 * temporal + 0.05 * classification
    return total, {"pixel": pixel, "texture": texture, "temporal": temporal, "classification": classification}


@torch.inference_mode()
def evaluate(loader, model, device):
    margins = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    sums = {name: 0.0 for name in ("baseline", "router_ungated", "pixel_oracle", "block_oracle", "highpass_baseline", "highpass_router_ungated")}
    margin_sums = {margin: 0.0 for margin in margins}
    count = hp_count = 0
    winner = np.zeros(3, dtype=np.float64)
    weight_sum = np.zeros(3, dtype=np.float64)
    model.eval()
    for candidates, context, actions, target in loader:
        candidates, context, actions, target = prepare(candidates, context, actions, target, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, weights = model(candidates, context, actions)
        prediction = prediction.float()
        error = (candidates - target[:, :, None]).abs().mean(dim=3)
        choose = error.argmin(dim=2)
        oracle = torch.gather(candidates, 2, choose[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        smooth = F.avg_pool2d(error.flatten(0, 1), 5, stride=1, padding=2, count_include_pad=False).unflatten(0, target.shape[:2])
        block_choose = smooth.argmin(dim=2)
        block_oracle = torch.gather(candidates, 2, block_choose[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        sums["baseline"] += float((candidates[:, :, 0] - target).abs().sum())
        sums["router_ungated"] += float((prediction - target).abs().sum())
        sums["pixel_oracle"] += float((oracle - target).abs().sum())
        sums["block_oracle"] += float((block_oracle - target).abs().sum())
        hp_target = highpass(target)
        sums["highpass_baseline"] += float((highpass(candidates[:, :, 0]) - hp_target).abs().sum())
        sums["highpass_router_ungated"] += float((highpass(prediction) - hp_target).abs().sum())
        irasim_weight, irasim_index = weights[:, :, 1:].max(dim=2)
        log_margin = irasim_weight.clamp_min(1e-8).log() - weights[:, :, 0].clamp_min(1e-8).log()
        for margin in margins:
            choice = torch.where(log_margin > margin, irasim_index + 1, torch.zeros_like(irasim_index))
            routed = torch.gather(candidates, 2, choice[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            margin_sums[margin] += float((routed - target).abs().sum())
        count += target.numel(); hp_count += target.numel()
        winner += np.asarray([(choose == index).sum().item() for index in range(3)])
        weight_sum += weights.sum(dim=(0, 1, 3, 4)).float().cpu().numpy()
    metrics = {name: value / (count if not name.startswith("highpass") else hp_count) * 255.0 for name, value in sums.items()}
    margin_metrics = {str(margin): value / count * 255.0 for margin, value in margin_sums.items()}
    best_margin = min(margins, key=lambda margin: margin_sums[margin])
    metrics["router"] = margin_metrics[str(best_margin)]
    metrics["routing_margin"] = best_margin
    metrics["margin_rgb_mae"] = margin_metrics
    metrics["relative_improvement"] = (metrics["baseline"] - metrics["router"]) / metrics["baseline"]
    metrics["oracle_relative_improvement"] = (metrics["baseline"] - metrics["pixel_oracle"]) / metrics["baseline"]
    metrics["winner_fraction"] = (winner / winner.sum()).tolist()
    metrics["mean_weight"] = (weight_sum / weight_sum.sum()).tolist()
    return metrics


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidate-cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.candidate_cache) != 3:
        raise ValueError("exactly three --candidate-cache arguments are required")
    candidates, names = load_candidates(args.candidate_cache)
    action_values = []
    for name in names:
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            action_values.append(window["future_actions"].astype(np.float32))
    action_values = np.concatenate(action_values, axis=0)
    action_mean = action_values.mean(axis=0)
    action_std = np.maximum(action_values.std(axis=0), 1e-6)
    episodes = sorted({episode(name) for name in names})
    rng = np.random.default_rng(args.seed)
    dev_count = max(1, round(len(episodes) * args.dev_fraction))
    dev_episodes = set(rng.choice(episodes, dev_count, replace=False).tolist())
    train_indices = [index for index, name in enumerate(names) if episode(name) not in dev_episodes]
    dev_indices = [index for index, name in enumerate(names) if episode(name) in dev_episodes]
    if not train_indices or not dev_indices:
        raise ValueError("episode split produced an empty partition")
    common = {"batch_size": args.batch_size, "num_workers": 2, "pin_memory": True}
    train_loader = DataLoader(RouterDataset(Path(args.windows), candidates, names, train_indices, action_mean, action_std), shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed), **common)
    dev_loader = DataLoader(RouterDataset(Path(args.windows), candidates, names, dev_indices, action_mean, action_std), shuffle=False, **common)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = MultiCandidateRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []
    initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial})
    best_improvement = initial["relative_improvement"]
    config = vars(args) | {"train_samples": len(train_indices), "dev_samples": len(dev_indices), "dev_episodes": sorted(dev_episodes)}
    atomic_save({"format": "track2-multi-candidate-router-v1", "state_dict": model.state_dict(), "config": config, "step": 0, "metrics": initial}, output / "best.pt")
    print(json.dumps(history[-1]), flush=True)
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        candidates_batch, context, actions, target = prepare(*batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, weights = model(candidates_batch, context, actions)
            loss, components = router_loss(prediction, weights, candidates_batch, context, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss), **{key: float(value) for key, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics})
            if metrics["relative_improvement"] > best_improvement:
                best_improvement = metrics["relative_improvement"]
                atomic_save({"format": "track2-multi-candidate-router-v1", "state_dict": model.state_dict(), "config": config, "step": step, "metrics": metrics}, output / "best.pt")
            print(json.dumps(history[-1]), flush=True)
    (output / "training_manifest.json").write_text(json.dumps(config | {"best_relative_improvement": best_improvement, "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
