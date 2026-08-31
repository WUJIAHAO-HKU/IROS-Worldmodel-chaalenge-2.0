#!/usr/bin/env python3
"""Train the dual-arm four-class full gripper structure head."""

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
from wam_pipeline.contact_structure_head_v135 import ContactStructureHeadV135
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from train_contact_occlusion_head_v13 import contact_score, episode, images
from train_contact_occlusion_head_v131 import active_arm


class StructureDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std
        self.arms = []
        for index in indices:
            with np.load(windows / names[index], allow_pickle=False) as window:
                arm, _ = active_arm(window["history_actions"], window["future_actions"])
            self.arms.append(arm)

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]
        y0, y1, x0, x1 = CONTACT_REGION
        with np.load(self.windows / name, allow_pickle=False) as window:
            context = window["context_frames"][-1, y0:y1, x0:x1].copy()
            target = window["target_frames"][:, y0:y1, x0:x1].copy()
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        action = (action - self.action_mean[arm]) / self.action_std[arm]
        parent = self.parent[index, :, y0:y1, x0:x1]
        source_label = structure_semantic_mask(context[None])[0]
        parent_label = structure_semantic_mask(parent)
        source_semantic = np.stack([source_label == label for label in (1, 2, 3)]).astype(np.float32)
        parent_semantic = np.stack(
            [parent_label == label for label in (1, 2, 3)], axis=1
        ).astype(np.float32)
        return parent, context, action, arm, structure_semantic_mask(target), source_semantic, parent_semantic, name


def loss_function(logits: torch.Tensor, target: torch.Tensor, motion_weight: float = 0.0,
                  temporal_weight: float = 0.015,
                  false_positive_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    weights = logits.new_tensor((0.12, 1.5, 3.2, 2.6))
    ce_map = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1),
                             weight=weights, reduction="none").unflatten(0, target.shape[:2])
    if motion_weight:
        changed = torch.zeros_like(target, dtype=logits.dtype)
        changed[:, 1:] = (target[:, 1:] != target[:, :-1]).to(logits.dtype)
        ce = (ce_map * (1 + motion_weight * changed)).sum() / (
            (1 + motion_weight * changed).sum() + 1e-6
        )
    else:
        ce = ce_map.mean()
    probability = logits.softmax(dim=2); dice_losses = []; precision_losses = []
    for label in (1, 2, 3):
        prediction = probability[:, :, label]; truth = (target == label).to(prediction.dtype)
        intersection = (prediction * truth).sum(dim=(-1, -2))
        dice = (2 * intersection + 1) / (
            prediction.sum(dim=(-1, -2)) + truth.sum(dim=(-1, -2)) + 1
        )
        dice_losses.append(1 - dice.mean())
        # Tversky with alpha>beta directly targets the old-position residue:
        # a confident structure pixel where the structure has moved/been
        # occluded costs more than a missed boundary pixel.
        false_positive = (prediction * (1 - truth)).sum(dim=(-1, -2))
        false_negative = ((1 - prediction) * truth).sum(dim=(-1, -2))
        tversky = (intersection + 1) / (
            intersection + 0.72 * false_positive + 0.28 * false_negative + 1
        )
        precision_losses.append(1 - tversky.mean())
    temporal = (probability[:, 1:] - probability[:, :-1]).abs().mean()
    precision = 0.8 * precision_losses[1] + 1.2 * precision_losses[2]
    total = (ce + 0.55 * dice_losses[0] + 1.1 * dice_losses[1] + dice_losses[2]
             + false_positive_weight * precision + temporal_weight * temporal)
    return total, {"cross_entropy": ce, "bottle_dice": dice_losses[0],
                   "black_dice": dice_losses[1], "grey_dice": dice_losses[2],
                   "structure_precision": precision, "temporal": temporal}


@torch.inference_mode()
def evaluate(loader, model, device) -> dict:
    model.eval()
    totals = {arm: {"i": np.zeros(4), "u": np.zeros(4), "truth": np.zeros(4),
                    "pred": np.zeros(4), "frame_i": np.zeros((2, 8)),
                    "frame_u": np.zeros((2, 8)), "samples": 0} for arm in (0, 1)}
    for parent, last, actions, arms, target, source_semantic, parent_semantic, _ in loader:
        prediction = model(
            last.permute(0, 3, 1, 2).to(device).float() / 255,
            images(parent, device), actions.to(device).float(), arms.to(device).long(),
            source_semantic.to(device).float(), parent_semantic.to(device).float()
        ).argmax(dim=2)
        target = target.to(device).long()
        for sample, arm in enumerate(arms.tolist()):
            state = totals[arm]; state["samples"] += 1
            for label in (1, 2, 3):
                pred, truth = prediction[sample] == label, target[sample] == label
                state["i"][label] += int((pred & truth).sum())
                state["u"][label] += int((pred | truth).sum())
                state["truth"][label] += int(truth.sum()); state["pred"][label] += int(pred.sum())
            for layer, label in enumerate((2, 3)):
                for time in range(8):
                    pred = prediction[sample, time] == label; truth = target[sample, time] == label
                    state["frame_i"][layer, time] += int((pred & truth).sum())
                    state["frame_u"][layer, time] += int((pred | truth).sum())
    result = {}
    for arm, state in totals.items():
        for name, label in (("bottle", 1), ("black", 2), ("grey", 3)):
            prefix = f"arm{arm}_{name}"
            result[prefix + "_iou"] = state["i"][label] / max(state["u"][label], 1)
            result[prefix + "_recall"] = state["i"][label] / max(state["truth"][label], 1)
            result[prefix + "_precision"] = state["i"][label] / max(state["pred"][label], 1)
        result[f"arm{arm}_black_frame_iou"] = (
            state["frame_i"][0] / np.maximum(state["frame_u"][0], 1)
        ).tolist()
        result[f"arm{arm}_grey_frame_iou"] = (
            state["frame_i"][1] / np.maximum(state["frame_u"][1], 1)
        ).tolist()
        result[f"arm{arm}_samples"] = state["samples"]
    result["worst_arm_structure_iou"] = min(
        result[f"arm{arm}_{layer}_iou"] for arm in (0, 1) for layer in ("black", "grey")
    )
    result["worst_final_frame_iou"] = min(
        result[f"arm{arm}_{layer}_frame_iou"][-1]
        for arm in (0, 1) for layer in ("black", "grey")
    )
    return result


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--arm0-dev-episode", default="episode36"); parser.add_argument("--arm1-dev-episode", default="episode47")
    parser.add_argument("--action-dropout", type=float, default=0.15)
    parser.add_argument("--semantic-dropout", type=float, default=0.35)
    parser.add_argument("--motion-weight", type=float, default=0.0)
    parser.add_argument("--temporal-weight", type=float, default=0.015)
    parser.add_argument("--false-positive-weight", type=float, default=0.0)
    parser.add_argument("--parent-rgb-dropout", type=float, default=0.0)
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
    train = StructureDataset(Path(args.windows), parent, names, train_indices, action_mean, action_std)
    dev = StructureDataset(Path(args.windows), parent, names, dev_indices, action_mean, action_std)
    counts = np.bincount(train.arms, minlength=2); weights = [1 / counts[arm] for arm in train.arms]
    sampler = WeightedRandomSampler(weights, len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    device = torch.device(args.device); model = ContactStructureHeadV135(args.base_channels).to(device)
    if args.initialize:
        initial_checkpoint = torch.load(args.initialize, map_location="cpu", weights_only=False)
        model.load_state_dict(initial_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); history = []
    initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True); best = initial["worst_arm_structure_iou"]
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        parent_batch, last, actions, arms, target, source_semantic, parent_semantic, _ = batch
        parent_batch = images(parent_batch, device); last = last.permute(0, 3, 1, 2).to(device).float() / 255
        actions = actions.to(device).float(); arms = arms.to(device).long(); target = target.to(device).long()
        source_semantic = source_semantic.to(device).float(); parent_semantic = parent_semantic.to(device).float()
        if args.action_dropout:
            drop = torch.rand(len(actions), device=device) < args.action_dropout
            actions = torch.where(drop[:, None, None], torch.zeros_like(actions), actions)
        if args.semantic_dropout:
            drop = torch.rand(len(actions), device=device) < args.semantic_dropout
            parent_semantic = parent_semantic.clone()
            parent_semantic[drop, :, 1:] = 0
        if args.parent_rgb_dropout:
            # Force a subset of sequences to derive motion from the recurrent
            # state/actions instead of copying the parent's stale black/grey
            # geometry. Bottle semantics remain available as an occlusion cue.
            drop = torch.rand(len(actions), device=device) < args.parent_rgb_dropout
            parent_batch = torch.where(
                drop[:, None, None, None, None],
                last[:, None].expand_as(parent_batch), parent_batch
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(last, parent_batch, actions, arms, source_semantic, parent_semantic)
            loss, parts = loss_function(logits, target, args.motion_weight, args.temporal_weight,
                                        args.false_positive_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
            value = {"format": "track2-contact-structure-head-v13.5", "model_version": "v13.5",
                     "state_dict": model.state_dict(), "step": step, "metrics": metrics,
                     "base_channels": args.base_channels, "action_mean": torch.from_numpy(action_mean),
                     "action_std": torch.from_numpy(action_std), "dev_episodes": sorted(dev_episodes)}
            atomic_save(value, output / "latest.pt")
            if metrics["worst_arm_structure_iou"] > best:
                best = metrics["worst_arm_structure_iou"]; atomic_save(value, output / "best.pt")
    manifest = {"format": "track2-contact-structure-v13.5-training", "steps": args.steps,
                "selected_windows": len(selected), "train_windows": len(train), "dev_windows": len(dev),
                "train_arm_counts": counts.tolist(), "dev_arm_counts": np.bincount(dev.arms, minlength=2).tolist(),
                "train_episodes": sorted({episode(names[i]) for i in train_indices}),
                "dev_episodes": sorted(dev_episodes), "motion_weight": args.motion_weight,
                "temporal_weight": args.temporal_weight,
                "false_positive_weight": args.false_positive_weight,
                "parent_rgb_dropout": args.parent_rgb_dropout,
                "initialize": args.initialize,
                "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
