#!/usr/bin/env python3
"""Train either object-regeneration expert against a deployment-matched parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import (
    CanonicalArmTextureV11, REGIONS, _observed_beam_mask, _observed_logo_mask, _warp,
    geometry_descriptor,
)
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.object_regeneration_experts_v160 import (
    GlyphAtlasProjectionExpert, GripperInstanceFlowExpert, parameter_counts,
    render_atlas_glyph, render_gripper_instances,
)
from train_contact_occlusion_head_v13 import contact_score, episode
from train_contact_occlusion_head_v131 import active_arm


def rgb_tensor(value: np.ndarray) -> torch.Tensor:
    axes = (2, 0, 1) if value.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(value.transpose(axes).copy()).float().div_(255)


def mask_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(value[:, None].copy()).float() if value.ndim == 3 else torch.from_numpy(value[None].copy()).float()


def clean_glyph(frame: np.ndarray) -> np.ndarray:
    mask = _observed_logo_mask(frame)
    if not mask.any():
        return frame
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(frame, mask * 255, 3.0, cv2.INPAINT_TELEA)


def actions_for(path: Path) -> tuple[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as window:
        return active_arm(window["history_actions"], window["future_actions"])


def action_statistics(windows: Path, names: list[str], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[np.ndarray]] = [[], []]
    for index in indices:
        arm, action = actions_for(windows / names[index]); values[arm].append(action)
    mean = np.zeros((2, 7), np.float32); std = np.ones((2, 7), np.float32)
    for arm in (0, 1):
        merged = np.concatenate(values[arm], 0)
        mean[arm] = merged.mean(0); std[arm] = np.maximum(merged.std(0), 1e-4)
    return mean, std


class AtlasAligner:
    """Retrieve and align a train-only atlas frame using observation geometry only."""

    def __init__(self, path: str | Path) -> None:
        self.renderer = CanonicalArmTextureV11(path)

    def __call__(self, context: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, bool]:
        y0, y1, x0, x1 = REGIONS[side]; query = context[y0:y1, x0:x1]
        descriptor = geometry_descriptor(context, side)
        indices = np.flatnonzero(self.renderer.sides == side)
        distance = ((self.renderer.descriptors[indices] - descriptor[None]) ** 2).mean((1, 2, 3))
        best = None
        for index in indices[np.argsort(distance)[:8]]:
            source = self.renderer.frames[index, y0:y1, x0:x1]
            aligned = self.renderer._align(query, source)
            # The conservative atlas renderer rejects many valid wrist poses
            # when the dark beam touches the wrist.  Its observed-beam PCA/ECC
            # initializer remains geometry-only and gives a safe canonical
            # transform for those cases.
            if aligned is None:
                aligned = self.renderer._align_observed(query, source)
            if aligned is None:
                continue
            correlation, matrix = aligned
            score = correlation - 0.08 * float(distance[np.flatnonzero(indices == index)[0]])
            if best is None or score > best[0]:
                best = (score, int(index), matrix)
        if best is None:
            observed_mask = _observed_logo_mask(query)
            return query, observed_mask, False
        _, index, matrix = best
        source = self.renderer.frames[index, y0:y1, x0:x1]
        aligned_rgb = _warp(source.astype(np.float32) / 255.0, matrix, cv2.INTER_CUBIC)
        aligned_mask = _warp(self.renderer.text_masks[index].astype(np.float32), matrix, cv2.INTER_LINEAR)
        aligned_rgb = np.round(np.clip(aligned_rgb, 0, 1) * 255).astype(np.uint8)
        return aligned_rgb, (aligned_mask > 0.15).astype(np.uint8), True


class GlyphDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 mean: np.ndarray, std: np.ndarray, atlas_rgb: np.ndarray,
                 atlas_mask: np.ndarray, atlas_ok: np.ndarray, atlas_positions: dict[int, int]) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.mean, self.std = mean, std
        self.atlas_rgb, self.atlas_mask, self.atlas_ok = atlas_rgb, atlas_mask, atlas_ok
        self.atlas_positions = atlas_positions
        self.arms = [actions_for(windows / names[index])[0] for index in indices]

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]; arm, action = actions_for(self.windows / name)
        side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
        with np.load(self.windows / name, allow_pickle=False) as window:
            target = window["target_frames"][:, y0:y1, x0:x1]
        parent = self.parent[index, :, y0:y1, x0:x1]
        position = self.atlas_positions[index]
        atlas_rgb, atlas_mask = self.atlas_rgb[position], self.atlas_mask[position]
        clean = np.stack([clean_glyph(frame) for frame in parent])
        parent_beam = np.stack([_observed_beam_mask(frame) for frame in parent])
        parent_glyph = np.stack([_observed_logo_mask(frame) for frame in parent])
        target_glyph = np.stack([_observed_logo_mask(frame) for frame in target])
        action = (action - self.mean[arm]) / self.std[arm]
        return (rgb_tensor(atlas_rgb), rgb_tensor(parent), rgb_tensor(clean), rgb_tensor(target),
                torch.from_numpy(action).float(), arm, mask_tensor(atlas_mask),
                mask_tensor(parent_beam), mask_tensor(parent_glyph), mask_tensor(target_glyph),
                bool(self.atlas_ok[position]), name)


class GripperDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 mean: np.ndarray, std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.mean, self.std = mean, std
        self.arms = [actions_for(windows / names[index])[0] for index in indices]

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]; y0, y1, x0, x1 = CONTACT_REGION
        with np.load(self.windows / name, allow_pickle=False) as window:
            source = window["context_frames"][-1, y0:y1, x0:x1]
            target = window["target_frames"][:, y0:y1, x0:x1]
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        parent = self.parent[index, :, y0:y1, x0:x1]
        source_label = structure_semantic_mask(source[None])[0]
        parent_label = structure_semantic_mask(parent); target_label = structure_semantic_mask(target)
        action = (action - self.mean[arm]) / self.std[arm]
        return (rgb_tensor(source), rgb_tensor(parent), rgb_tensor(target), torch.from_numpy(action).float(), arm,
                mask_tensor(source_label == 2), mask_tensor(source_label == 3),
                mask_tensor(parent_label == 2), mask_tensor(parent_label == 3),
                mask_tensor(parent_label == 1), torch.from_numpy(target_label.copy()).long(), name)


def highpass(value: torch.Tensor) -> torch.Tensor:
    return value - F.avg_pool3d(value, (1, 5, 5), 1, (0, 2, 2))


def soft_dice(probability: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    intersection = (probability * truth).sum((-1, -2, -3))
    return 1 - ((2 * intersection + 1) /
                (probability.sum((-1, -2, -3)) + truth.sum((-1, -2, -3)) + 1)).mean()


def glyph_forward(model, batch, device, strength: float = 1.0):
    atlas, parent, clean, target, actions, arms, atlas_mask, parent_beam, parent_glyph, truth, atlas_ok, names = batch
    tensors = [atlas, parent, clean, target, actions, atlas_mask, parent_beam, parent_glyph, truth]
    atlas, parent, clean, target, actions, atlas_mask, parent_beam, parent_glyph, truth = [v.to(device).float() for v in tensors]
    arms = arms.to(device).long()
    flow, visibility, edit = model(atlas, parent, actions, arms, atlas_mask, parent_beam, parent_glyph)
    output, alpha, warped = render_atlas_glyph(parent, atlas, atlas_mask, parent_beam,
                                               flow, visibility, edit, strength,
                                               parent_clean=clean)
    return output, alpha, warped, flow, visibility, edit, parent, target, truth, arms, names


def glyph_loss(model, batch, device):
    output, alpha, warped, flow, visibility, edit, parent, target, truth, _, _ = glyph_forward(model, batch, device)
    valid = (truth.sum((-1, -2, -3)) >= 4).float()
    dice = soft_dice(alpha, truth)
    probability = alpha.float().clamp(1e-5, 1 - 1e-5)
    bce = (-(truth * probability.log() + (1 - truth) * (1 - probability).log())
           * (1 + 30 * truth)).mean()
    region = F.max_pool3d(torch.maximum(truth, alpha.detach()), (1, 9, 9), 1, (0, 4, 4))
    rgb = ((output - target).abs() * region).sum() / (3 * region.sum() + 1)
    edge = ((highpass(output) - highpass(target)).abs() * region).sum() / (3 * region.sum() + 1)
    identity = ((output - parent).abs() * (1 - region)).mean()
    flow_tv = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    acceleration = (flow[:, 2:] - 2 * flow[:, 1:-1] + flow[:, :-2]).abs().mean()
    visibility_truth = valid[:, :, None, None, None].expand_as(visibility)
    visibility_loss = F.binary_cross_entropy_with_logits(visibility, visibility_truth)
    edit_truth = F.max_pool3d(truth, (1, 5, 5), 1, (0, 2, 2))
    edit_loss = (F.binary_cross_entropy_with_logits(edit, edit_truth, reduction="none")
                 * (1 + 15 * edit_truth)).mean()
    total = (dice + 0.06 * bce + 4 * rgb + 2 * edge + 2 * identity + 0.03 * flow_tv
             + 0.02 * acceleration + 0.1 * visibility_loss + 0.25 * edit_loss)
    return total, {"dice": dice, "bce": bce, "rgb": rgb, "edge": edge,
                   "identity": identity, "flow_tv": flow_tv, "visibility": visibility_loss,
                   "edit": edit_loss}


def gripper_forward(model, batch, device, strength: float = 1.0):
    source, parent, target, actions, arms, source_black, source_grey, parent_black, parent_grey, parent_bottle, truth, names = batch
    tensors = [source, parent, target, actions, source_black, source_grey, parent_black, parent_grey, parent_bottle]
    source, parent, target, actions, source_black, source_grey, parent_black, parent_grey, parent_bottle = [v.to(device).float() for v in tensors]
    arms, truth = arms.to(device).long(), truth.to(device).long()
    semantic, flow, visibility, replacement, edit = model(
        source, parent, actions, arms, source_black, source_grey, parent_black, parent_grey, parent_bottle)
    output, probability, alpha, warped = render_gripper_instances(
        parent, source, source_black, source_grey, parent_black, parent_grey,
        semantic, flow, visibility, replacement, edit, strength)
    return output, probability, alpha, warped, semantic, flow, visibility, edit, parent, target, truth, arms, names


def gripper_loss(model, batch, device):
    output, probability, alpha, warped, semantic, flow, visibility, edit, parent, target, truth, _, _ = gripper_forward(model, batch, device)
    semantic_truth = torch.zeros_like(truth); semantic_truth[truth == 2] = 1; semantic_truth[truth == 3] = 2
    changed = torch.zeros_like(semantic_truth, dtype=torch.float32)
    changed[:, 1:] = (semantic_truth[:, 1:] != semantic_truth[:, :-1]).float()
    ce_map = F.cross_entropy(semantic.flatten(0, 1), semantic_truth.flatten(0, 1),
                             weight=semantic.new_tensor((0.25, 3.5, 2.5)), reduction="none").unflatten(0, semantic_truth.shape[:2])
    ce = (ce_map * (1 + 3 * changed)).mean()
    black, grey = (semantic_truth == 1).float()[:, :, None], (semantic_truth == 2).float()[:, :, None]
    dice_black = soft_dice(probability[:, :, 1:2], black)
    dice_grey = soft_dice(probability[:, :, 2:3], grey)
    target_structure = torch.maximum(black, grey)
    parent_black = batch[7].to(device).float(); parent_grey = batch[8].to(device).float()
    parent_semantic = torch.zeros_like(semantic_truth)
    parent_semantic[parent_black[:, :, 0] > .5] = 1
    parent_semantic[parent_grey[:, :, 0] > .5] = 2
    region = F.max_pool3d(torch.maximum(target_structure, alpha.detach()), (1, 11, 11), 1, (0, 5, 5))
    rgb = ((output - target).abs() * region).sum() / (3 * region.sum() + 1)
    edge = ((highpass(output) - highpass(target)).abs() * region).sum() / (3 * region.sum() + 1)
    identity = ((output - parent).abs() * (1 - region)).mean()
    flow_tv = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    temporal = (probability[:, 1:] - probability[:, :-1]).abs().mean()
    changed = (semantic_truth != parent_semantic)[:, :, None].float()
    photometric_error = (parent - target).abs().mean(2, keepdim=True)
    editable_union = F.max_pool3d(torch.maximum(target_structure, torch.maximum(parent_black, parent_grey)),
                                  (1, 9, 9), 1, (0, 4, 4))
    edit_truth = torch.maximum(changed, (photometric_error > .05).float() * editable_union)
    edit_loss = (F.binary_cross_entropy_with_logits(edit, edit_truth, reduction="none")
                 * (1 + 8 * edit_truth)).mean()
    total = (ce + dice_black + 0.8 * dice_grey + 4 * rgb + 2 * edge + 2 * identity
             + 0.03 * flow_tv + 0.01 * temporal + 0.3 * edit_loss)
    return total, {"ce": ce, "dice_black": dice_black, "dice_grey": dice_grey,
                   "rgb": rgb, "edge": edge, "identity": identity, "flow_tv": flow_tv,
                   "edit": edit_loss}


@torch.inference_mode()
def evaluate(loader, model, device, expert: str) -> dict:
    model.eval(); totals = {arm: {"before": 0., "after": 0., "pixels": 0., "i": 0, "u": 0, "count": 0} for arm in (0, 1)}
    horizon = {time: {"before": 0., "after": 0., "pixels": 0.} for time in range(8)}
    for batch in loader:
        if expert == "glyph":
            output, alpha, _, _, _, _, parent, target, truth, arms, _ = glyph_forward(model, batch, device)
            region = F.max_pool3d(truth, (1, 9, 9), 1, (0, 4, 4)); prediction = alpha >= .35; truth_bool = truth > .5
        else:
            output, probability, alpha, _, _, _, _, _, parent, target, truth_label, arms, _ = gripper_forward(model, batch, device)
            truth_bool = (truth_label == 2)[:, :, None] | (truth_label == 3)[:, :, None]
            prediction = probability[:, :, 1:].sum(2, keepdim=True) >= .5
            region = F.max_pool3d(truth_bool.float(), (1, 9, 9), 1, (0, 4, 4))
        for sample, arm in enumerate(arms.tolist()):
            state = totals[arm]; pixels = float(3 * region[sample].sum())
            state["before"] += float(((parent[sample] - target[sample]).abs() * region[sample]).sum())
            state["after"] += float(((output[sample] - target[sample]).abs() * region[sample]).sum())
            state["pixels"] += pixels; state["count"] += 1
            state["i"] += int((prediction[sample] & truth_bool[sample]).sum())
            state["u"] += int((prediction[sample] | truth_bool[sample]).sum())
            for time in range(8):
                pixel = float(3 * region[sample, time].sum()); horizon[time]["pixels"] += pixel
                horizon[time]["before"] += float(((parent[sample, time] - target[sample, time]).abs() * region[sample, time]).sum())
                horizon[time]["after"] += float(((output[sample, time] - target[sample, time]).abs() * region[sample, time]).sum())
    result = {}; ratios = []
    for arm, state in totals.items():
        before = state["before"] / max(state["pixels"], 1); after = state["after"] / max(state["pixels"], 1)
        result[f"arm{arm}_samples"] = state["count"]; result[f"arm{arm}_roi_mae_parent"] = before
        result[f"arm{arm}_roi_mae_expert"] = after; result[f"arm{arm}_roi_gain"] = (before - after) / max(before, 1e-9)
        result[f"arm{arm}_instance_iou"] = state["i"] / max(state["u"], 1)
        if state["count"]: ratios.append(after / max(before, 1e-9))
    result["horizon"] = []
    for time, state in horizon.items():
        before = state["before"] / max(state["pixels"], 1); after = state["after"] / max(state["pixels"], 1)
        result["horizon"].append({"horizon": time + 1, "parent": before, "expert": after,
                                  "gain": (before - after) / max(before, 1e-9)})
    result["selection_ratio"] = float(max(ratios))
    return result


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=("glyph", "gripper"), required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--atlas"); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1600); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24); parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--init-checkpoint", help="Optional same-expert v16 checkpoint for parent-chain fine-tuning")
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--arm0-dev-episode", default="episode36"); parser.add_argument("--arm1-dev-episode", default="episode47")
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; names = cache["windows"].astype(str).tolist()
    windows = Path(args.windows); selected, errors = [], []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            context, target = window["context_frames"][-1], window["target_frames"]
            arm, _ = active_arm(window["history_actions"], window["future_actions"])
        if args.expert == "glyph":
            side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
            mask = np.stack([_observed_logo_mask(frame[y0:y1, x0:x1]) for frame in target])
            keep = _observed_logo_mask(context[y0:y1, x0:x1]).sum() >= 8 and mask.sum() >= 8
            error = float(np.abs(parent[index, :, y0:y1, x0:x1].astype(np.float32) - target[:, y0:y1, x0:x1]).mean())
        else:
            y0, y1, x0, x1 = CONTACT_REGION; score = contact_score(target[:, y0:y1, x0:x1])
            keep = score[0] >= 2 and score[1] >= 1000 and score[2] >= 80
            p = structure_semantic_mask(parent[index, :, y0:y1, x0:x1]); t = structure_semantic_mask(target[:, y0:y1, x0:x1])
            error = float(((p == 2) != (t == 2)).mean() + ((p == 3) != (t == 3)).mean())
        if keep: selected.append(index); errors.append(error)
        if (index + 1) % 200 == 0: print(json.dumps({"scan": index + 1, "selected": len(selected)}), flush=True)
    dev_episodes = {args.arm0_dev_episode, args.arm1_dev_episode}
    train_indices = [i for i in selected if episode(names[i]) not in dev_episodes]
    dev_indices = [i for i in selected if episode(names[i]) in dev_episodes]
    mean, std = action_statistics(windows, names, train_indices)
    atlas_coverage = None
    if args.expert == "glyph":
        if not args.atlas: raise ValueError("--atlas is required for glyph training")
        aligner = AtlasAligner(args.atlas); atlas_rgbs, atlas_masks, atlas_oks = [], [], []
        for number, index in enumerate(selected):
            with np.load(windows / names[index], allow_pickle=False) as window: context = window["context_frames"][-1]
            arm, _ = actions_for(windows / names[index]); side = "left" if arm == 0 else "right"
            rgb, mask, ok = aligner(context, side); atlas_rgbs.append(rgb); atlas_masks.append(mask); atlas_oks.append(ok)
            if (number + 1) % 100 == 0: print(json.dumps({"atlas_aligned": number + 1, "accepted": sum(atlas_oks)}), flush=True)
        atlas_rgbs, atlas_masks, atlas_oks = np.stack(atlas_rgbs), np.stack(atlas_masks), np.asarray(atlas_oks)
        positions = {index: position for position, index in enumerate(selected)}
        dataset_args = (atlas_rgbs, atlas_masks, atlas_oks, positions)
        atlas_coverage = float(atlas_oks.mean())
        dataset_type, model_type, loss_function = GlyphDataset, GlyphAtlasProjectionExpert, glyph_loss
    else:
        dataset_args = (); dataset_type, model_type, loss_function = GripperDataset, GripperInstanceFlowExpert, gripper_loss
    train = dataset_type(windows, parent, names, train_indices, mean, std, *dataset_args)
    dev = dataset_type(windows, parent, names, dev_indices, mean, std, *dataset_args)
    counts = np.bincount(train.arms, minlength=2); selected_error = {index: error for index, error in zip(selected, errors)}
    error_values = np.asarray([selected_error[i] for i in train_indices], np.float64)
    error_values = error_values / max(np.median(error_values), 1e-6)
    weights = [float((1 / max(counts[arm], 1)) * np.clip(0.5 + error_values[pos], 0.5, 4.0))
               for pos, arm in enumerate(train.arms)]
    sampler = WeightedRandomSampler(weights, len(train), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    device = torch.device(args.device); model = model_type(args.base_channels).to(device)
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial_checkpoint.get("expert") != args.expert:
            raise ValueError("init checkpoint expert does not match --expert")
        if int(initial_checkpoint["base_channels"]) != args.base_channels:
            raise ValueError("init checkpoint base_channels does not match")
        model.load_state_dict(initial_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []; initial = evaluate(dev_loader, model, device, args.expert); history.append({"step": 0, **initial})
    print(json.dumps(history[-1]), flush=True); best = initial["selection_ratio"]
    initial_state = {"format": f"track2-object-regeneration-experts-v16.0-{args.expert}",
                     "expert": args.expert, "state_dict": model.state_dict(), "step": 0,
                     "metrics": initial, "base_channels": args.base_channels,
                     "action_mean": torch.from_numpy(mean), "action_std": torch.from_numpy(std),
                     "dev_episodes": sorted(dev_episodes), "atlas": args.atlas,
                     "atlas_coverage": atlas_coverage}
    atomic_save(initial_state, output / "latest.pt"); atomic_save(initial_state, output / "best.pt")
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, parts = loss_function(model, batch, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss), **{k: float(v) for k, v in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device, args.expert); history.append({"step": step, **metrics}); print(json.dumps(history[-1]), flush=True)
            checkpoint = {"format": f"track2-object-regeneration-experts-v16.0-{args.expert}", "expert": args.expert,
                          "state_dict": model.state_dict(), "step": step, "metrics": metrics,
                          "base_channels": args.base_channels, "action_mean": torch.from_numpy(mean),
                          "action_std": torch.from_numpy(std), "dev_episodes": sorted(dev_episodes),
                          "atlas": args.atlas, "atlas_coverage": atlas_coverage}
            atomic_save(checkpoint, output / "latest.pt")
            if metrics["selection_ratio"] < best: best = metrics["selection_ratio"]; atomic_save(checkpoint, output / "best.pt")
    manifest = {"format": "track2-object-regeneration-experts-v16.0-training", "expert": args.expert,
                "steps": args.steps, "selected_windows": len(selected), "train_windows": len(train), "dev_windows": len(dev),
                "train_arm_counts": counts.tolist(), "dev_arm_counts": np.bincount(dev.arms, minlength=2).tolist(),
                "parent_cache": args.parent_cache, "atlas": args.atlas, "atlas_coverage": atlas_coverage,
                "init_checkpoint": args.init_checkpoint,
                "parameter_counts": parameter_counts(args.base_channels), "hard_case_sampling": True, "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
