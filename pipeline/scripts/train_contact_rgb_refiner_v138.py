#!/usr/bin/env python3
"""Train the recurrent contact RGB warp/fusion refiner on balanced arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_rgb_refiner_v138 import ContactRGBRefinerV138
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from train_contact_occlusion_head_v13 import contact_score, episode, images
from train_contact_occlusion_head_v131 import active_arm


class RGBDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std; self.arms = []
        for index in indices:
            with np.load(windows / names[index], allow_pickle=False) as window:
                arm, _ = active_arm(window["history_actions"], window["future_actions"])
            self.arms.append(arm)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]; y0, y1, x0, x1 = CONTACT_REGION
        with np.load(self.windows / name, allow_pickle=False) as window:
            last = window["context_frames"][-1, y0:y1, x0:x1].copy()
            target = window["target_frames"][:, y0:y1, x0:x1].copy()
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        parent = self.parent[index, :, y0:y1, x0:x1]
        action = (action - self.action_mean[arm]) / self.action_std[arm]
        source_label = structure_semantic_mask(last[None])[0]
        parent_label = structure_semantic_mask(parent)
        target_label = structure_semantic_mask(target)
        source_semantic = np.stack([source_label == label for label in (1, 2, 3)]).astype(np.float32)
        parent_semantic = np.stack([parent_label == label for label in (1, 2, 3)], axis=1).astype(np.float32)
        return parent, last, action, arm, target, target_label, source_semantic, parent_semantic, name


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return (flat - F.avg_pool2d(flat, 5, stride=1, padding=2,
                                count_include_pad=False)).unflatten(0, value.shape[:2])


def training_loss(result: dict, last: torch.Tensor, target: torch.Tensor,
                  labels: torch.Tensor, motion_weight: float = 4.0,
                  structure_weight: float = 3.0,
                  semantic_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    prediction = result["prediction"]
    previous = torch.cat((last[:, None], target[:, :-1]), dim=1)
    motion = (target - previous).abs().mean(dim=2, keepdim=True)
    structure = ((labels == 2) | (labels == 3))[:, :, None]
    dark = target.mean(dim=2, keepdim=True) < 0.32
    weight = 1 + motion_weight * (motion >= 0.025) + structure_weight * structure + 1.5 * dark
    pixel = (weight * (prediction - target).abs()).sum() / (weight.sum() * 3)
    texture = (weight * (highpass(prediction) - highpass(target)).abs()).sum() / (weight.sum() * 3)
    edge_x = ((prediction[..., 1:] - prediction[..., :-1]) -
              (target[..., 1:] - target[..., :-1])).abs().mean()
    edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
              (target[..., 1:, :] - target[..., :-1, :])).abs().mean()
    edge = 0.5 * (edge_x + edge_y)
    flow = result["flow"]
    flow_smooth = (flow[..., 1:] - flow[..., :-1]).abs().mean()
    flow_smooth += (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    flow_regularizer = flow.abs().mean() / 28 + 0.25 * flow_smooth / 28
    correction_regularizer = result["correction"].abs().mean()
    # Continuous colour-to-structure scores add direct geometric pressure to
    # the RGB branch. This prevents a low-amplitude colour tweak from winning
    # while the old black/grey silhouette remains in place.
    luminance = prediction.mean(dim=2)
    chroma = prediction.max(dim=2).values - prediction.min(dim=2).values
    black_score = torch.sigmoid((0.32 - luminance) * 22) * torch.exp(-6 * chroma)
    grey_score = (torch.sigmoid((luminance - 0.32) * 18) *
                  torch.sigmoid((0.80 - luminance) * 18) * torch.exp(-6 * chroma))
    semantic_losses = []
    for score, label in ((black_score, 2), (grey_score, 3)):
        truth = (labels == label).to(score.dtype)
        intersection = (score * truth).sum(dim=(-1, -2))
        dice = (2 * intersection + 1) / (score.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2)) + 1)
        semantic_losses.append(1 - dice.mean())
    semantic = semantic_losses[0] + semantic_losses[1]
    total = (pixel + 0.55 * texture + 0.35 * edge + semantic_weight * semantic
             + 0.002 * flow_regularizer + 0.02 * correction_regularizer)
    return total, {"pixel": pixel, "texture": texture, "edge": edge,
                   "semantic": semantic,
                   "flow_regularizer": flow_regularizer,
                   "correction_regularizer": correction_regularizer}


@torch.inference_mode()
def evaluate(loader, model, device) -> dict:
    model.eval(); totals = {arm: {"parent": 0.0, "prediction": 0.0, "structure_parent": 0.0,
                                  "structure_prediction": 0.0, "pixels": 0,
                                  "structure_pixels": 0, "frame_parent": np.zeros(8),
                                  "frame_prediction": np.zeros(8), "frame_pixels": np.zeros(8),
                                  "samples": 0} for arm in (0, 1)}
    for parent, last, actions, arms, target, labels, source_semantic, parent_semantic, _ in loader:
        parent = images(parent, device); target = images(target, device)
        last = last.permute(0, 3, 1, 2).to(device).float() / 255
        actions, arms = actions.to(device).float(), arms.to(device).long()
        labels = labels.to(device); source_semantic = source_semantic.to(device).float()
        parent_semantic = parent_semantic.to(device).float()
        prediction = model(last, parent, actions, arms, source_semantic, parent_semantic).clamp(0, 1)
        for sample, arm in enumerate(arms.tolist()):
            state = totals[arm]; state["samples"] += 1
            parent_error = (parent[sample] - target[sample]).abs()
            prediction_error = (prediction[sample] - target[sample]).abs()
            structure = ((labels[sample] == 2) | (labels[sample] == 3))[:, None]
            state["parent"] += float(parent_error.sum()); state["prediction"] += float(prediction_error.sum())
            state["pixels"] += target[sample].numel()
            state["structure_parent"] += float((parent_error * structure).sum())
            state["structure_prediction"] += float((prediction_error * structure).sum())
            state["structure_pixels"] += int(structure.sum()) * 3
            state["frame_parent"] += parent_error.mean(dim=(1, 2, 3)).cpu().numpy()
            state["frame_prediction"] += prediction_error.mean(dim=(1, 2, 3)).cpu().numpy()
            state["frame_pixels"] += 1
    result = {}
    for arm, state in totals.items():
        parent = 255 * state["parent"] / max(state["pixels"], 1)
        refined = 255 * state["prediction"] / max(state["pixels"], 1)
        prefix = f"arm{arm}"; result[prefix + "_samples"] = state["samples"]
        result[prefix + "_parent_mae"] = parent; result[prefix + "_refined_mae"] = refined
        result[prefix + "_improvement_percent"] = 100 * (parent - refined) / max(parent, 1e-6)
        result[prefix + "_structure_parent_mae"] = 255 * state["structure_parent"] / max(state["structure_pixels"], 1)
        result[prefix + "_structure_refined_mae"] = 255 * state["structure_prediction"] / max(state["structure_pixels"], 1)
        result[prefix + "_frame_parent_mae"] = (255 * state["frame_parent"] / state["frame_pixels"]).tolist()
        result[prefix + "_frame_refined_mae"] = (255 * state["frame_prediction"] / state["frame_pixels"]).tolist()
    result["worst_arm_improvement_percent"] = min(result["arm0_improvement_percent"],
                                                   result["arm1_improvement_percent"])
    for arm in (0, 1):
        before = result[f"arm{arm}_structure_parent_mae"]
        after = result[f"arm{arm}_structure_refined_mae"]
        result[f"arm{arm}_structure_improvement_percent"] = 100 * (before - after) / max(before, 1e-6)
    result["worst_arm_structure_improvement_percent"] = min(
        result["arm0_structure_improvement_percent"], result["arm1_structure_improvement_percent"]
    )
    return result


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--arm0-dev-episode", default="episode36"); parser.add_argument("--arm1-dev-episode", default="episode47")
    parser.add_argument("--motion-weight", type=float, default=4.0)
    parser.add_argument("--structure-weight", type=float, default=3.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--initialize")
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    selected, metadata = [], {}; y0, y1, x0, x1 = CONTACT_REGION
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            score = contact_score(window["target_frames"][:, y0:y1, x0:x1])
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        if score[0] >= 2 and score[1] >= 1000 and score[2] >= 80:
            selected.append(index); metadata[index] = (arm, action)
    dev_episodes = {args.arm0_dev_episode, args.arm1_dev_episode}
    train_indices = [i for i in selected if episode(names[i]) not in dev_episodes]
    dev_indices = [i for i in selected if episode(names[i]) in dev_episodes]
    action_mean = np.zeros((2, 7), np.float32); action_std = np.ones((2, 7), np.float32)
    for arm in (0, 1):
        values = np.concatenate([metadata[i][1] for i in train_indices if metadata[i][0] == arm])
        action_mean[arm] = values.mean(0); action_std[arm] = np.maximum(values.std(0), 1e-4)
    train = RGBDataset(Path(args.windows), parent, names, train_indices, action_mean, action_std)
    dev = RGBDataset(Path(args.windows), parent, names, dev_indices, action_mean, action_std)
    counts = np.bincount(train.arms, minlength=2); sample_weights = [1 / counts[arm] for arm in train.arms]
    sampler = WeightedRandomSampler(sample_weights, len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    device = torch.device(args.device); model = ContactRGBRefinerV138(args.base_channels).to(device)
    if args.initialize:
        initial_checkpoint = torch.load(args.initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(initial_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); history = []
    initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True); best = initial["worst_arm_structure_improvement_percent"]
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        parent_batch, last, actions, arms, target, labels, source_semantic, parent_semantic, _ = batch
        parent_batch = images(parent_batch, device); target = images(target, device)
        last = last.permute(0, 3, 1, 2).to(device).float() / 255
        actions, arms, labels = actions.to(device).float(), arms.to(device).long(), labels.to(device)
        source_semantic = source_semantic.to(device).float(); parent_semantic = parent_semantic.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(last, parent_batch, actions, arms, source_semantic, parent_semantic, return_details=True)
            loss, parts = training_loss(result, last, target, labels, args.motion_weight,
                                        args.structure_weight, args.semantic_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
            value = {"format": "track2-contact-rgb-refiner-v13.8", "model_version": "v13.8",
                     "state_dict": model.state_dict(), "step": step, "metrics": metrics,
                     "base_channels": args.base_channels, "action_mean": torch.from_numpy(action_mean),
                     "action_std": torch.from_numpy(action_std), "dev_episodes": sorted(dev_episodes)}
            atomic_save(value, output / "latest.pt")
            if metrics["worst_arm_structure_improvement_percent"] > best:
                best = metrics["worst_arm_structure_improvement_percent"]
                atomic_save(value, output / "best.pt")
    manifest = {"format": "track2-contact-rgb-refiner-v13.8-training", "steps": args.steps,
                "selected_windows": len(selected), "train_windows": len(train), "dev_windows": len(dev),
                "train_arm_counts": counts.tolist(), "dev_arm_counts": np.bincount(dev.arms, minlength=2).tolist(),
                "dev_episodes": sorted(dev_episodes), "motion_weight": args.motion_weight,
                "structure_weight": args.structure_weight, "semantic_weight": args.semantic_weight,
                "initialize": args.initialize, "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
