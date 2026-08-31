#!/usr/bin/env python3
"""Train the v13 recurrent contact/occlusion head on episode-disjoint data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_layer_v12 import bottle_mask, gripper_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION, ContactOcclusionHeadV13


def episode(name: str) -> str:
    match = re.search(r"episode\d+", name)
    if not match:
        raise ValueError(f"no episode in {name}")
    return match.group(0)


def semantic_mask(frames: np.ndarray) -> np.ndarray:
    labels = []
    for frame in frames:
        bottle = bottle_mask(frame) > 0
        gripper = gripper_mask(frame, bottle.astype(np.uint8)) > 0
        label = bottle.astype(np.uint8)
        label[gripper] = 2
        labels.append(label)
    return np.stack(labels)


def contact_score(target: np.ndarray) -> tuple[int, int, int]:
    labels = semantic_mask(target)
    bottle = labels == 1
    gripper = labels == 2
    touching = 0
    for bmask, gmask in zip(bottle, gripper):
        near = cv2.dilate(bmask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        touching += int(np.logical_and(near, gmask).any())
    return touching, int(bottle.sum()), int(gripper.sum())


class ContactDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            y0, y1, x0, x1 = CONTACT_REGION
            context = window["context_frames"][-1, y0:y1, x0:x1].copy()
            target = window["target_frames"][:, y0:y1, x0:x1].copy()
            actions = (window["future_actions"].copy() - self.action_mean) / self.action_std
        return self.parent[index, :, y0:y1, x0:x1], context, actions, semantic_mask(target), name


def images(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def loss_function(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    weights = logits.new_tensor((0.18, 1.8, 3.8))
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), weight=weights)
    probability = logits.softmax(dim=2)
    dice_losses = []
    for label in (1, 2):
        prediction = probability[:, :, label]
        truth = (target == label).to(prediction.dtype)
        intersection = (prediction * truth).sum(dim=(-1, -2))
        dice = (2 * intersection + 1) / (prediction.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2)) + 1)
        dice_losses.append(1 - dice.mean())
    temporal = (probability[:, 1:] - probability[:, :-1]).abs().mean()
    total = ce + 0.7 * dice_losses[0] + 1.1 * dice_losses[1] + 0.025 * temporal
    return total, {"cross_entropy": ce, "bottle_dice_loss": dice_losses[0],
                   "gripper_dice_loss": dice_losses[1], "temporal": temporal}


@torch.inference_mode()
def evaluate(loader, model, device) -> dict:
    model.eval()
    intersections = np.zeros(3, np.float64); unions = np.zeros(3, np.float64)
    recalls_n = np.zeros(3, np.float64); recalls_d = np.zeros(3, np.float64)
    correct = total = sequence_present = sequence_count = 0
    frame_intersections = np.zeros(8); frame_unions = np.zeros(8)
    for parent, last, actions, target, _ in loader:
        parent = images(parent, device)
        last = last.permute(0, 3, 1, 2).to(device).float().div(255.0)
        target = target.to(device).long(); actions = actions.to(device).float()
        prediction = model(last, parent, actions).argmax(dim=2)
        correct += int((prediction == target).sum()); total += target.numel()
        for label in (1, 2):
            pred, truth = prediction == label, target == label
            intersections[label] += int((pred & truth).sum())
            unions[label] += int((pred | truth).sum())
            recalls_n[label] += int((pred & truth).sum()); recalls_d[label] += int(truth.sum())
        pg, tg = prediction == 2, target == 2
        for time in range(8):
            frame_intersections[time] += int((pg[:, time] & tg[:, time]).sum())
            frame_unions[time] += int((pg[:, time] | tg[:, time]).sum())
        predicted_all = pg.flatten(2).any(dim=2).all(dim=1)
        target_all = tg.flatten(2).any(dim=2).all(dim=1)
        sequence_present += int((predicted_all & target_all).sum()); sequence_count += int(target_all.sum())
    return {
        "pixel_accuracy": correct / max(total, 1),
        "bottle_iou": intersections[1] / max(unions[1], 1),
        "gripper_iou": intersections[2] / max(unions[2], 1),
        "bottle_recall": recalls_n[1] / max(recalls_d[1], 1),
        "gripper_recall": recalls_n[2] / max(recalls_d[2], 1),
        "gripper_all_8_frames_recall": sequence_present / max(sequence_count, 1),
        "gripper_frame_iou": (frame_intersections / np.maximum(frame_unions, 1)).tolist(),
    }


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--dev-episodes", default="episode32,episode47")
    parser.add_argument("--minimum-touching-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    selected = []
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            y0, y1, x0, x1 = CONTACT_REGION
            score = contact_score(window["target_frames"][:, y0:y1, x0:x1])
        if score[0] >= args.minimum_touching_frames and score[1] >= 1000 and score[2] >= 80:
            selected.append(index)
    dev_episodes = set(args.dev_episodes.split(","))
    train_indices = [i for i in selected if episode(names[i]) not in dev_episodes]
    dev_indices = [i for i in selected if episode(names[i]) in dev_episodes]
    if not train_indices or not dev_indices:
        raise ValueError(f"empty split: train={len(train_indices)} dev={len(dev_indices)}")
    action_values = []
    for index in train_indices:
        with np.load(Path(args.windows) / names[index], allow_pickle=False) as window:
            action_values.append(window["future_actions"].copy())
    action_values = np.concatenate(action_values)
    action_mean = action_values.mean(axis=0).astype(np.float32)
    action_std = np.maximum(action_values.std(axis=0), 1e-4).astype(np.float32)
    train = ContactDataset(Path(args.windows), parent, names, train_indices, action_mean, action_std)
    dev = ContactDataset(Path(args.windows), parent, names, dev_indices, action_mean, action_std)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        persistent_workers=True, generator=generator)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    device = torch.device(args.device); model = ContactOcclusionHeadV13(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []; initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True); best = initial["gripper_iou"]; iterator = iter(loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: parent_batch, last, actions, target, _ = next(iterator)
        except StopIteration: iterator = iter(loader); parent_batch, last, actions, target, _ = next(iterator)
        parent_batch = images(parent_batch, device)
        last = last.permute(0, 3, 1, 2).to(device).float().div(255.0)
        actions = actions.to(device).float(); target = target.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(last, parent_batch, actions); loss, parts = loss_function(logits, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
            value = {"format": "track2-contact-occlusion-head-v13", "state_dict": model.state_dict(),
                     "step": step, "metrics": metrics, "base_channels": args.base_channels,
                     "action_mean": torch.from_numpy(action_mean), "action_std": torch.from_numpy(action_std),
                     "dev_episodes": sorted(dev_episodes)}
            atomic_save(value, output / "latest.pt")
            if metrics["gripper_iou"] > best:
                best = metrics["gripper_iou"]; atomic_save(value, output / "best.pt")
    manifest = {"format": "track2-contact-occlusion-v13-training", "steps": args.steps,
                "selected_windows": len(selected), "train_windows": len(train), "dev_windows": len(dev),
                "train_episodes": sorted({episode(names[i]) for i in train_indices}),
                "dev_episodes": sorted(dev_episodes), "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

