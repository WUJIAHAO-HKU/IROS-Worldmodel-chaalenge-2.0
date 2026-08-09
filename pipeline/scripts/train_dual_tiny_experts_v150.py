#!/usr/bin/env python3
"""Train either frozen-parent specialist in the independent v15.0 experiment."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import (
    REGIONS, _observed_beam_mask, _observed_logo_mask,
)
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.dual_tiny_experts_v150 import (
    TinyBlackGripperExpert, TinyGlyphMotionExpert, parameter_counts,
    render_black_gripper, render_glyph,
)


SIZE = 96


def episode(name: str) -> str:
    match = re.search(r"episode\d+", name)
    if not match:
        raise ValueError(f"no episode in {name}")
    return match.group(0)


def active_arm(history: np.ndarray, future: np.ndarray) -> tuple[int, np.ndarray]:
    actions = np.concatenate((history, future), axis=0)
    delta = np.abs(np.diff(actions, axis=0))
    arm = int(np.asarray((delta[:, :7].mean(), delta[:, 7:].mean())).argmax())
    return arm, future[:, arm * 7:(arm + 1) * 7].copy()


def contact_score(target: np.ndarray) -> tuple[int, int, int]:
    labels = structure_semantic_mask(target)
    bottle, gripper = labels == 1, labels == 2
    touching = 0
    for bottle_mask, gripper_mask in zip(bottle, gripper):
        near = cv2.dilate(bottle_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        touching += int(np.logical_and(near, gripper_mask).any())
    return touching, int(bottle.sum()), int(gripper.sum())


def _resize_rgb(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3:
        return cv2.resize(value, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.stack([_resize_rgb(frame) for frame in value])


def _resize_mask(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return cv2.resize(value.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([_resize_mask(frame) for frame in value])


def _rgb_tensor(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3:
        return value.transpose(2, 0, 1).astype(np.float32) / 255.0
    return value.transpose(0, 3, 1, 2).astype(np.float32) / 255.0


def _mask_tensor(value: np.ndarray) -> np.ndarray:
    return value[:, None].astype(np.float32) if value.ndim == 3 else value[None].astype(np.float32)


def _clean_glyph(frame: np.ndarray) -> np.ndarray:
    existing = _observed_logo_mask(frame)
    if not existing.any():
        return frame
    clear = cv2.dilate(existing, np.ones((3, 3), np.uint8))
    return cv2.inpaint(frame, clear * 255, 2.0, cv2.INPAINT_TELEA)


def _actions_and_arm(path: Path) -> tuple[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as window:
        return active_arm(window["history_actions"], window["future_actions"])


def action_statistics(windows: Path, names: list[str], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[np.ndarray]] = [[], []]
    for index in indices:
        arm, action = _actions_and_arm(windows / names[index])
        values[arm].append(action)
    mean = np.zeros((2, 7), np.float32); std = np.ones((2, 7), np.float32)
    for arm in (0, 1):
        merged = np.concatenate(values[arm], 0)
        mean[arm] = merged.mean(0); std[arm] = np.maximum(merged.std(0), 1e-4)
    return mean, std


class GlyphDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std
        self.arms = [_actions_and_arm(windows / names[index])[0] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            context = window["context_frames"][-1]
            target = window["target_frames"]
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
        source = context[y0:y1, x0:x1]
        target = target[:, y0:y1, x0:x1]
        parent = self.parent[index, :, y0:y1, x0:x1]
        source_glyph = _observed_logo_mask(source)
        source_beam = _observed_beam_mask(source)
        parent_beam = np.stack([_observed_beam_mask(frame) for frame in parent])
        target_glyph = np.stack([_observed_logo_mask(frame) for frame in target])
        parent_clean = np.stack([_clean_glyph(frame) for frame in parent])
        action = (action - self.action_mean[arm]) / self.action_std[arm]
        return (_rgb_tensor(_resize_rgb(source)), _rgb_tensor(_resize_rgb(parent)),
                _rgb_tensor(_resize_rgb(parent_clean)), _rgb_tensor(_resize_rgb(target)),
                action.astype(np.float32), arm,
                _mask_tensor(_resize_mask(source_glyph)), _mask_tensor(_resize_mask(source_beam)),
                _mask_tensor(_resize_mask(parent_beam)), _mask_tensor(_resize_mask(target_glyph)), name)


class GripperDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std
        self.arms = [_actions_and_arm(windows / names[index])[0] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]
        y0, y1, x0, x1 = CONTACT_REGION
        with np.load(self.windows / name, allow_pickle=False) as window:
            context = window["context_frames"][-1, y0:y1, x0:x1]
            target = window["target_frames"][:, y0:y1, x0:x1]
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        parent = self.parent[index, :, y0:y1, x0:x1]
        source_label = structure_semantic_mask(context[None])[0]
        parent_label = structure_semantic_mask(parent)
        target_label = structure_semantic_mask(target)
        action = (action - self.action_mean[arm]) / self.action_std[arm]
        return (_rgb_tensor(_resize_rgb(context)), _rgb_tensor(_resize_rgb(parent)),
                _rgb_tensor(_resize_rgb(target)), action.astype(np.float32), arm,
                _mask_tensor(_resize_mask(source_label == 2)),
                _mask_tensor(_resize_mask(parent_label == 2)),
                _mask_tensor(_resize_mask(parent_label == 1)),
                _mask_tensor(_resize_mask(target_label == 2)), name)


def glyph_loss(model, batch, device, strength: float = 1.0):
    source, parent, clean, target, actions, arms, source_glyph, source_beam, parent_beam, target_glyph, _ = batch
    values = [source, parent, clean, target, actions, source_glyph, source_beam, parent_beam, target_glyph]
    source, parent, clean, target, actions, source_glyph, source_beam, parent_beam, target_glyph = [
        value.to(device, non_blocking=True).float() for value in values
    ]
    arms = arms.to(device, non_blocking=True).long()
    matrices, confidence = model(source, parent, actions, arms, source_glyph, source_beam, parent_beam)
    rendered, alpha = render_glyph(clean, source, source_glyph, parent_beam, matrices, confidence, strength)
    truth = target_glyph
    valid = (truth.sum(dim=(-1, -2, -3)) >= 4).float()
    intersection = (alpha * truth).sum(dim=(-1, -2, -3))
    dice = 1 - ((2 * intersection + 1) / (alpha.sum(dim=(-1, -2, -3))
                                             + truth.sum(dim=(-1, -2, -3)) + 1))
    dice = (dice * valid).sum() / (valid.sum() + 1e-6)
    positive_weight = truth.new_tensor(25.0)
    # The rendered alpha is a product of visibility, geometry and beam support,
    # so there is no single equivalent logit.  Evaluate BCE explicitly in fp32
    # to keep it safe under CUDA autocast.
    alpha_probability = alpha.float().clamp(1e-5, 1 - 1e-5)
    truth_probability = truth.float()
    bce_map = -(truth_probability * alpha_probability.log()
                + (1 - truth_probability) * (1 - alpha_probability).log())
    bce_map = bce_map * (1 + positive_weight * truth)
    bce = (bce_map.mean(dim=(-1, -2, -3)) * valid).sum() / (valid.sum() + 1e-6)
    region = F.max_pool3d(torch.maximum(truth, alpha.detach()), (1, 7, 7), 1, (0, 3, 3))
    rgb = ((rendered - target).abs() * region).sum() / (3 * region.sum() + 1e-6)
    rendered_hp = rendered - F.avg_pool3d(rendered, (1, 5, 5), 1, (0, 2, 2))
    target_hp = target - F.avg_pool3d(target, (1, 5, 5), 1, (0, 2, 2))
    highpass = ((rendered_hp - target_hp).abs() * region).sum() / (3 * region.sum() + 1e-6)
    confidence_probability = confidence.float().clamp(1e-5, 1 - 1e-5)
    confidence_loss = -(valid * confidence_probability.log()
                        + (1 - valid) * (1 - confidence_probability).log()).mean()
    temporal = (matrices[:, 1:] - matrices[:, :-1]).abs().mean()
    identity = ((rendered - clean).abs() * (1 - parent_beam)).mean()
    total = dice + 0.08 * bce + 3.0 * rgb + 1.5 * highpass + 0.2 * confidence_loss + 0.05 * temporal + identity
    return total, {"dice": dice, "bce": bce, "rgb": rgb, "highpass": highpass,
                   "confidence": confidence_loss, "temporal": temporal, "identity": identity}


def gripper_loss(model, batch, device, strength: float = 1.0):
    source, parent, target, actions, arms, source_black, parent_black, parent_bottle, target_black, _ = batch
    values = [source, parent, target, actions, source_black, parent_black, parent_bottle, target_black]
    source, parent, target, actions, source_black, parent_black, parent_bottle, target_black = [
        value.to(device, non_blocking=True).float() for value in values
    ]
    arms = arms.to(device, non_blocking=True).long()
    logits, residual = model(source, parent, actions, arms, source_black, parent_black, parent_bottle)
    rendered, probability, support = render_black_gripper(
        parent, source_black, parent_black, logits, residual, strength
    )
    changed = torch.zeros_like(target_black); changed[:, 1:] = (
        target_black[:, 1:] != target_black[:, :-1]
    ).float()
    bce_map = F.binary_cross_entropy_with_logits(logits, target_black, reduction="none")
    weight = 1 + 5 * target_black + 4 * changed
    bce = (bce_map * weight).sum() / weight.sum().clamp_min(1)
    intersection = (probability * target_black).sum(dim=(-1, -2, -3))
    false_positive = (probability * (1 - target_black)).sum(dim=(-1, -2, -3))
    false_negative = ((1 - probability) * target_black).sum(dim=(-1, -2, -3))
    dice = 1 - ((2 * intersection + 1) / (probability.sum(dim=(-1, -2, -3))
                                             + target_black.sum(dim=(-1, -2, -3)) + 1)).mean()
    tversky = 1 - ((intersection + 1) / (intersection + 0.75 * false_positive
                                         + 0.25 * false_negative + 1)).mean()
    region = F.max_pool3d(torch.maximum(target_black, parent_black), (1, 9, 9), 1, (0, 4, 4))
    rgb = ((rendered - target).abs() * region).sum() / (3 * region.sum() + 1e-6)
    rendered_hp = rendered - F.avg_pool3d(rendered, (1, 5, 5), 1, (0, 2, 2))
    target_hp = target - F.avg_pool3d(target, (1, 5, 5), 1, (0, 2, 2))
    edge = ((rendered_hp - target_hp).abs() * region).sum() / (3 * region.sum() + 1e-6)
    temporal = (probability[:, 1:] - probability[:, :-1]).abs().mean()
    total = bce + dice + 0.8 * tversky + 3.0 * rgb + edge + 0.02 * temporal
    return total, {"bce": bce, "dice": dice, "tversky": tversky, "rgb": rgb,
                   "edge": edge, "temporal": temporal}


@torch.inference_mode()
def evaluate_glyph(loader, model, device, strength: float = 1.0) -> dict:
    model.eval(); totals = {arm: {"before": 0.0, "after": 0.0, "pixels": 0.0,
                                 "crop_before": 0.0, "crop_after": 0.0, "count": 0} for arm in (0, 1)}
    for batch in loader:
        source, parent, clean, target, actions, arms, source_glyph, source_beam, parent_beam, target_glyph, _ = batch
        source, parent, clean, target, actions, source_glyph, source_beam, parent_beam, target_glyph = [
            value.to(device).float() for value in (source, parent, clean, target, actions, source_glyph,
                                                    source_beam, parent_beam, target_glyph)
        ]
        matrix, confidence = model(source, parent, actions, arms.to(device).long(), source_glyph,
                                   source_beam, parent_beam)
        output, _ = render_glyph(clean, source, source_glyph, parent_beam, matrix, confidence, strength)
        region = F.max_pool3d(target_glyph, (1, 7, 7), 1, (0, 3, 3))
        for sample, arm in enumerate(arms.tolist()):
            state = totals[arm]; pixels = float(3 * region[sample].sum())
            state["before"] += float(((parent[sample] - target[sample]).abs() * region[sample]).sum())
            state["after"] += float(((output[sample] - target[sample]).abs() * region[sample]).sum())
            state["pixels"] += pixels
            state["crop_before"] += float((parent[sample] - target[sample]).abs().mean())
            state["crop_after"] += float((output[sample] - target[sample]).abs().mean())
            state["count"] += 1
    result = {}
    ratios = []
    for arm, state in totals.items():
        before = state["before"] / max(state["pixels"], 1); after = state["after"] / max(state["pixels"], 1)
        result[f"arm{arm}_samples"] = state["count"]
        result[f"arm{arm}_text_mae_parent"] = before; result[f"arm{arm}_text_mae_expert"] = after
        result[f"arm{arm}_text_gain"] = (before - after) / max(before, 1e-9)
        result[f"arm{arm}_crop_mae_parent"] = state["crop_before"] / max(state["count"], 1)
        result[f"arm{arm}_crop_mae_expert"] = state["crop_after"] / max(state["count"], 1)
        if state["count"]: ratios.append(after / max(before, 1e-9))
    result["selection_ratio"] = float(max(ratios))
    return result


@torch.inference_mode()
def evaluate_gripper(loader, model, device, strength: float = 1.0) -> dict:
    model.eval(); totals = {arm: {"i": 0, "u": 0, "truth": 0, "pred": 0,
                                 "before": 0.0, "after": 0.0, "pixels": 0.0,
                                 "crop_before": 0.0, "crop_after": 0.0, "count": 0} for arm in (0, 1)}
    for batch in loader:
        source, parent, target, actions, arms, source_black, parent_black, parent_bottle, target_black, _ = batch
        source, parent, target, actions, source_black, parent_black, parent_bottle, target_black = [
            value.to(device).float() for value in (source, parent, target, actions, source_black,
                                                    parent_black, parent_bottle, target_black)
        ]
        logits, residual = model(source, parent, actions, arms.to(device).long(), source_black,
                                 parent_black, parent_bottle)
        output, probability, _ = render_black_gripper(parent, source_black, parent_black,
                                                       logits, residual, strength)
        prediction = probability >= 0.5
        for sample, arm in enumerate(arms.tolist()):
            state = totals[arm]; truth = target_black[sample] > 0.5; pred = prediction[sample]
            state["i"] += int((truth & pred).sum()); state["u"] += int((truth | pred).sum())
            state["truth"] += int(truth.sum()); state["pred"] += int(pred.sum())
            region = F.max_pool3d(target_black[sample:sample + 1], (1, 7, 7), 1, (0, 3, 3))[0]
            pixels = float(3 * region.sum()); state["pixels"] += pixels
            state["before"] += float(((parent[sample] - target[sample]).abs() * region).sum())
            state["after"] += float(((output[sample] - target[sample]).abs() * region).sum())
            state["crop_before"] += float((parent[sample] - target[sample]).abs().mean())
            state["crop_after"] += float((output[sample] - target[sample]).abs().mean())
            state["count"] += 1
    result = {}; scores = []
    for arm, state in totals.items():
        iou = state["i"] / max(state["u"], 1); before = state["before"] / max(state["pixels"], 1)
        after = state["after"] / max(state["pixels"], 1)
        result[f"arm{arm}_samples"] = state["count"]; result[f"arm{arm}_black_iou"] = iou
        result[f"arm{arm}_black_precision"] = state["i"] / max(state["pred"], 1)
        result[f"arm{arm}_black_recall"] = state["i"] / max(state["truth"], 1)
        result[f"arm{arm}_black_rgb_parent"] = before; result[f"arm{arm}_black_rgb_expert"] = after
        result[f"arm{arm}_black_rgb_gain"] = (before - after) / max(before, 1e-9)
        result[f"arm{arm}_crop_mae_parent"] = state["crop_before"] / max(state["count"], 1)
        result[f"arm{arm}_crop_mae_expert"] = state["crop_after"] / max(state["count"], 1)
        if state["count"]: scores.append(after / max(before, 1e-9) + (1 - iou))
    result["selection_score"] = float(max(scores))
    return result


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=("glyph", "gripper"), required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--arm0-dev-episode", default="episode36"); parser.add_argument("--arm1-dev-episode", default="episode47")
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]
        names = cache["windows"].astype(str).tolist()
    windows = Path(args.windows); selected, metadata = [], {}
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            context = window["context_frames"][-1]
            target = window["target_frames"]
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        if args.expert == "glyph":
            side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
            keep = _observed_logo_mask(context[y0:y1, x0:x1]).sum() >= 8
        else:
            y0, y1, x0, x1 = CONTACT_REGION
            score = contact_score(target[:, y0:y1, x0:x1])
            keep = score[0] >= 2 and score[1] >= 1000 and score[2] >= 80
        if keep:
            selected.append(index); metadata[index] = (arm, action)
        if (index + 1) % 200 == 0:
            print(json.dumps({"scan": index + 1, "selected": len(selected)}), flush=True)
    dev_episodes = {args.arm0_dev_episode, args.arm1_dev_episode}
    train_indices = [index for index in selected if episode(names[index]) not in dev_episodes]
    dev_indices = [index for index in selected if episode(names[index]) in dev_episodes]
    action_mean, action_std = action_statistics(windows, names, train_indices)
    dataset_type = GlyphDataset if args.expert == "glyph" else GripperDataset
    train = dataset_type(windows, parent, names, train_indices, action_mean, action_std)
    dev = dataset_type(windows, parent, names, dev_indices, action_mean, action_std)
    counts = np.bincount(train.arms, minlength=2); dev_counts = np.bincount(dev.arms, minlength=2)
    sample_weights = [1 / max(counts[arm], 1) for arm in train.arms]
    sampler = WeightedRandomSampler(sample_weights, len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    device = torch.device(args.device)
    model_type = TinyGlyphMotionExpert if args.expert == "glyph" else TinyBlackGripperExpert
    model = model_type(args.base_channels).to(device)
    loss_function = glyph_loss if args.expert == "glyph" else gripper_loss
    evaluate = evaluate_glyph if args.expert == "glyph" else evaluate_gripper
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []; initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True)
    key = "selection_ratio" if args.expert == "glyph" else "selection_score"
    best = initial[key]; iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, parts = loss_function(model, batch, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{name: float(value) for name, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)
            checkpoint = {"format": f"track2-dual-tiny-experts-v15.0-{args.expert}",
                          "expert": args.expert, "state_dict": model.state_dict(), "step": step,
                          "metrics": metrics, "base_channels": args.base_channels,
                          "action_mean": torch.from_numpy(action_mean),
                          "action_std": torch.from_numpy(action_std),
                          "dev_episodes": sorted(dev_episodes), "input_size": SIZE}
            atomic_save(checkpoint, output / "latest.pt")
            if metrics[key] < best:
                best = metrics[key]; atomic_save(checkpoint, output / "best.pt")
    manifest = {"format": "track2-dual-tiny-experts-v15.0-training", "expert": args.expert,
                "steps": args.steps, "selected_windows": len(selected),
                "train_windows": len(train), "dev_windows": len(dev),
                "train_arm_counts": counts.tolist(), "dev_arm_counts": dev_counts.tolist(),
                "train_episodes": sorted({episode(names[index]) for index in train_indices}),
                "dev_episodes": sorted(dev_episodes), "parameter_counts": parameter_counts(),
                "parent_cache": args.parent_cache, "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
