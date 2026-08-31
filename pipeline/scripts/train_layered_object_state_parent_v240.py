#!/usr/bin/env python3
"""Train v24 with explicit object layers and full eight-step supervision."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wam_pipeline.geometry_visibility_router_v172 import _inpaint_candidate, full_layer_masks
from wam_pipeline.layered_object_state_parent_v240 import (
    LayeredObjectStateParentV240, parameter_count,
)
from wam_pipeline.object_geometry_v170 import affine_from_landmarks, warp_layer
from train_contact_occlusion_head_v131 import active_arm
from train_geometry_visibility_router_v172 import load_pose, precompute_geometry


REGION = (72, 232, 30, 210)


def episode(name: str) -> int:
    match = re.match(r"episode(\d+)_", name)
    if match is None:
        raise ValueError(name)
    return int(match.group(1))


def load_actions(windows: Path, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions, arms = [], []
    for name in names:
        with np.load(windows / name, allow_pickle=False) as value:
            arm, action = active_arm(value["history_actions"], value["future_actions"])
        actions.append(action); arms.append(arm)
    return np.stack(actions).astype(np.float32), np.asarray(arms, np.int64)


def action_stats(actions: np.ndarray, arms: np.ndarray, indices: list[int]):
    mean = np.stack([actions[indices][arms[indices] == arm].mean((0, 1)) for arm in (0, 1)])
    std = np.stack([actions[indices][arms[indices] == arm].std((0, 1)) for arm in (0, 1)])
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def positive_layer_indices(context: np.ndarray, target: np.ndarray, arms: np.ndarray,
                           indices: list[int]) -> list[int]:
    """Windows with an observable source layer and at least one future label."""
    selected = []
    for order, index in enumerate(indices):
        side = "left" if int(arms[index]) == 0 else "right"
        source = full_layer_masks(context[index, -1], side)[:4].sum()
        future = sum(full_layer_masks(frame, side)[:4].sum() for frame in target[index])
        if source >= 20 and future >= 20:
            selected.append(index)
        if (order + 1) % 300 == 0:
            print(json.dumps({"label_scan": order + 1, "total": len(indices)}), flush=True)
    return selected


def make_example(index, parent, target, context, source_geometry, future_geometry, arms):
    side = "left" if int(arms[index]) == 0 else "right"
    transported, support, cleanup, cleanup_support, labels = [], [], [], [], []
    source_rgb = context[index, -1]
    source_layers = full_layer_masks(source_rgb, side)
    source_structure = source_layers[1:3].max(0).astype(bool)
    for time in range(8):
        matrix = affine_from_landmarks(source_geometry[index, 4], future_geometry[index, time])
        moved_rgb = warp_layer(source_rgb, matrix, cv2.INTER_CUBIC)
        moved_support = np.stack([
            warp_layer(layer, matrix, cv2.INTER_NEAREST) for layer in source_layers[:4]
        ])
        moved_union = moved_support.max(0).astype(bool)
        stale = source_structure & ~cv2.dilate(
            moved_union.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        transported.append(moved_rgb); support.append(moved_support)
        cleanup.append(_inpaint_candidate(parent[index, time], stale))
        cleanup_support.append(stale[None])
        labels.append(full_layer_masks(target[index, time], side)[:4])
    return (parent[index], target[index], np.stack(transported), np.stack(support),
            np.stack(cleanup), np.stack(cleanup_support), np.stack(labels))


def tensors(batch, actions, arms, indices, mean, std, poses, device):
    def rgb(values, position):
        return torch.from_numpy(np.stack([value[position] for value in values])
                                .transpose(0, 1, 4, 2, 3).copy()).to(device).float() / 255
    base = rgb(batch, 0); truth = rgb(batch, 1); transported = rgb(batch, 2)
    support = torch.from_numpy(np.stack([value[3] for value in batch])).to(device).float()
    cleanup = rgb(batch, 4)
    cleanup_support = torch.from_numpy(np.stack([value[5] for value in batch])).to(device).float()
    labels = torch.from_numpy(np.stack([value[6] for value in batch])).to(device).float()
    normalized = np.stack([(actions[i] - mean[arms[i]]) / std[arms[i]] for i in indices])
    action = torch.from_numpy(normalized).to(device).float()
    arm = torch.from_numpy(arms[indices]).to(device).long()
    pose = torch.from_numpy(poses[indices, :, :6] / 128 - 1).to(device).float()
    return base, truth, transported, support, cleanup, cleanup_support, labels, action, pose, arm


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return (flat - F.avg_pool2d(flat, 5, 1, 2, count_include_pad=False)).reshape_as(value)


def loss_terms(output, truth, base, labels, detail):
    object_support = labels.max(2, keepdim=True).values
    previous_truth = torch.cat((truth[:, :1], truth[:, :-1]), 1)
    motion = (truth - previous_truth).abs().mean(2, keepdim=True)
    motion = F.max_pool2d((motion > .012).flatten(0, 1).float(), 7, 1, 3).reshape_as(motion)
    weights = 1 + 4 * object_support + 1.5 * motion
    pixel = ((output - truth).abs() * weights).sum() / (weights.sum() * 3)
    parent_error = (base - truth).abs().mean(2)
    candidate_error = (detail["layer_rgbs"] - truth[:, :, None]).abs().mean(3)
    expanded_labels = F.max_pool2d(labels.flatten(0, 1), 5, 1, 2).reshape_as(labels)
    benefit = ((parent_error[:, :, None] - candidate_error) > (.5 / 255))
    benefit &= detail["warped_support"] > .25
    benefit &= expanded_labels > .5
    benefit = benefit.to(output.dtype).detach()
    gate_weight = 1 + 8 * benefit
    mask_loss = ((detail["soft_masks"] - benefit).square() * gate_weight).sum() / gate_weight.sum()
    intersection = (detail["soft_masks"] * benefit).sum((0, 1, 3, 4))
    denominator = detail["soft_masks"].sum((0, 1, 3, 4)) + benefit.sum((0, 1, 3, 4))
    dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
    semantic_intersection = (detail["warped_support"] * labels).sum((0, 1, 3, 4))
    semantic_denominator = detail["warped_support"].sum((0, 1, 3, 4)) + labels.sum((0, 1, 3, 4))
    semantic = (1 - (2 * semantic_intersection + 1) / (semantic_denominator + 1)).mean()
    detail_support = F.max_pool2d(object_support.flatten(0, 1), 5, 1, 2).reshape_as(object_support)
    texture = ((highpass(output) - highpass(truth)).abs() * (1 + 5 * detail_support)).sum()
    texture /= ((1 + 5 * detail_support).sum() * 3)
    target_delta = torch.cat((truth[:, :1], truth), 1).diff(dim=1)
    temporal = (torch.cat((truth[:, :1], output), 1).diff(dim=1) - target_delta).abs().mean()
    protected = 1 - F.max_pool2d(
        torch.maximum(object_support, support_from_detail(detail)).flatten(0, 1), 11, 1, 5
    ).reshape_as(object_support)
    protect = ((output - base).abs() * protected).sum() / (protected.sum() * 3).clamp_min(1)
    flow = detail["flows"]
    smooth = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    smooth += (flow[..., 1:] - flow[..., :-1]).abs().mean()
    magnitude = flow.abs().mean()
    loss = pixel + .45 * texture + .30 * temporal + .40 * mask_loss + .10 * dice + .05 * semantic
    loss = loss + .50 * protect + .003 * smooth + .001 * magnitude
    return loss, {"pixel": pixel, "texture": texture, "temporal": temporal,
                  "mask": mask_loss, "dice": dice, "semantic": semantic,
                  "benefit_fraction": benefit.mean(), "protect": protect, "flow": magnitude}


def support_from_detail(detail):
    # Predicted visible layer support is the recurrent object's deployable state.
    return detail["masks"].max(2, keepdim=True).values


@torch.inference_mode()
def evaluate(model, indices, arrays, geometry, actions, arms, stats, device,
             maximum_windows=0, workers=4, mask_threshold=.5, active_from=0,
             enabled_layers=(True, True, True, True)):
    parent, target, context = arrays; source_geometry, future_geometry = geometry
    selected = list(indices)
    if maximum_windows and len(selected) > maximum_windows:
        positions = np.linspace(0, len(selected) - 1, maximum_windows, dtype=int)
        selected = [selected[p] for p in positions]
    names = ("rgb", "texture", "temporal", "contact", "structure")
    before = {name: torch.zeros(8) for name in names}; after = {name: torch.zeros(8) for name in names}
    arm_values = {0: {name: [0., 0.] for name in names}, 1: {name: [0., 0.] for name in names}}
    arm_counts = {0: 0, 1: 0}
    for order, index in enumerate(selected):
        example = make_example(index, parent, target, context, source_geometry, future_geometry, arms)
        values = tensors([example], actions, arms, [index], *stats, future_geometry, device)
        base, truth, transported, support, cleanup, cleanup_support, labels, action, pose, arm = values
        output, detail = model(base, transported, support, cleanup, cleanup_support,
                               action, pose, arm, teacher_forcing=0.0,
                               mask_threshold=mask_threshold, active_from=active_from,
                               enabled_layers=enabled_layers)
        y0, y1, x0, x1 = REGION
        b = {
            "rgb": (base - truth).abs().mean((2, 3, 4)),
            "texture": (highpass(base) - highpass(truth)).abs().mean((2, 3, 4)),
            "temporal": (torch.cat((truth[:, :1], base), 1).diff(dim=1) -
                         torch.cat((truth[:, :1], truth), 1).diff(dim=1)).abs().mean((2, 3, 4)),
            "contact": (base[:, :, :, y0:y1, x0:x1] - truth[:, :, :, y0:y1, x0:x1]).abs().mean((2, 3, 4)),
        }
        a = {
            "rgb": (output - truth).abs().mean((2, 3, 4)),
            "texture": (highpass(output) - highpass(truth)).abs().mean((2, 3, 4)),
            "temporal": (torch.cat((truth[:, :1], output), 1).diff(dim=1) -
                         torch.cat((truth[:, :1], truth), 1).diff(dim=1)).abs().mean((2, 3, 4)),
            "contact": (output[:, :, :, y0:y1, x0:x1] - truth[:, :, :, y0:y1, x0:x1]).abs().mean((2, 3, 4)),
        }
        structure = labels[:, :, 1:3].max(2).values[:, :, None].expand(-1, -1, 3, -1, -1)
        b["structure"] = ((base - truth).abs() * structure).sum((2, 3, 4)) / structure.sum((2, 3, 4)).clamp_min(1)
        a["structure"] = ((output - truth).abs() * structure).sum((2, 3, 4)) / structure.sum((2, 3, 4)).clamp_min(1)
        arm_id = int(arms[index]); arm_counts[arm_id] += 1
        for name in names:
            before[name] += b[name][0].cpu(); after[name] += a[name][0].cpu()
            arm_values[arm_id][name][0] += float(b[name].mean())
            arm_values[arm_id][name][1] += float(a[name].mean())
        if (order + 1) % 32 == 0:
            print(json.dumps({"eval": order + 1, "total": len(selected)}), flush=True)
    result = {"windows": len(selected), "arms": {}}
    gains = []
    for name in names:
        p = 255 * before[name] / len(selected); v = 255 * after[name] / len(selected)
        frame_gain = 100 * (p - v) / p.clamp_min(1e-9)
        result[f"parent_{name}_mae"] = float(p.mean()); result[f"{name}_mae"] = float(v.mean())
        result[f"{name}_improvement_percent"] = float(100 * (p.mean() - v.mean()) / p.mean())
        result[f"frame_{name}_improvement_percent"] = frame_gain.tolist()
        gains.extend((float(frame_gain.min()), result[f"{name}_improvement_percent"]))
    for arm_id in (0, 1):
        record = {"windows": arm_counts[arm_id]}
        for name in names:
            p, v = arm_values[arm_id][name]
            gain = 100 * (p - v) / max(p, 1e-9)
            record[f"{name}_improvement_percent"] = gain; gains.append(gain)
        result["arms"][f"arm{arm_id}"] = record
    result["gate_score"] = min(gains)
    return result


def atomic_save(payload, path):
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(payload, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--pose-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--dev-episodes", default="36,47"); parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260809); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precompute-cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=.9)
    parser.add_argument("--active-from", type=int, default=0)
    parser.add_argument("--disable-beam", action="store_true")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(1)
    device = torch.device("cuda"); windows = Path(args.windows)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; target = cache["target"]; context = cache["context"]
        names = cache["windows"].astype(str)
    dev_episodes = {int(value) for value in args.dev_episodes.split(",") if value}
    train = [i for i, name in enumerate(names) if episode(name) not in dev_episodes]
    dev = [i for i, name in enumerate(names) if episode(name) in dev_episodes]
    precompute_path = Path(args.precompute_cache) if args.precompute_cache else None
    if precompute_path is not None and precompute_path.is_file():
        with np.load(precompute_path, allow_pickle=False) as cached:
            if not np.array_equal(cached["windows"].astype(str), names):
                raise ValueError("precompute cache window identity mismatch")
            source_geometry = cached["source_geometry"]; future_geometry = cached["future_geometry"]
            arms = cached["arms"]; actions = cached["actions"]
            positives = cached["positive_indices"].tolist()
        print(json.dumps({"loaded_precompute_cache": str(precompute_path)}), flush=True)
    else:
        pose_model = load_pose(args.pose_checkpoint, device)
        source_geometry, future_geometry, arms = precompute_geometry(windows, names, pose_model, device)
        actions, action_arms = load_actions(windows, names); assert np.array_equal(arms, action_arms)
        positives = positive_layer_indices(context, target, arms, train)
        if precompute_path is not None:
            precompute_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = precompute_path.with_suffix(f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, windows=names, source_geometry=source_geometry,
                                    future_geometry=future_geometry, arms=arms, actions=actions,
                                    positive_indices=np.asarray(positives, np.int64))
            os.replace(temporary, precompute_path)
    mean, std = action_stats(actions, arms, train); stats = (mean, std)
    by_arm = {arm: [i for i in train if arms[i] == arm] for arm in (0, 1)}
    positive_by_arm = {arm: [i for i in positives if arms[i] == arm] for arm in (0, 1)}
    if any(not positive_by_arm[arm] for arm in (0, 1)):
        raise ValueError(f"missing positive layer windows: {positive_by_arm}")
    print(json.dumps({"positive_layer_windows": {str(arm): len(values)
                      for arm, values in positive_by_arm.items()}}), flush=True)
    model = LayeredObjectStateParentV240(args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=args.workers); history = []; best = -1e9; start_step = 0
    arrays = (parent, target, context); geometry = (source_geometry, future_geometry)
    enabled_layers = (not args.disable_beam, True, True, True)
    state_path = output / "training_state.pt"
    if args.resume and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]); scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["step"]); best = float(saved["best"]); history = saved["history"]
        rng.bit_generator.state = saved["rng_state"]
        print(json.dumps({"resumed_step": start_step, "best_gate_score": best}), flush=True)
    for step in range(start_step + 1, args.steps + 1):
        indices = []
        for offset in range(args.batch_size):
            arm_id = (step + offset) % 2
            pool = positive_by_arm[arm_id] if rng.random() < .90 else by_arm[arm_id]
            indices.append(int(rng.choice(pool)))
        futures = [executor.submit(make_example, index, *arrays, *geometry, arms) for index in indices]
        batch = [future.result() for future in futures]
        values = tensors(batch, actions, arms, indices, mean, std, future_geometry, device)
        base, truth, transported, support, cleanup, cleanup_support, labels, action, pose, arm = values
        teacher_forcing = max(.10, .80 * (1 - step / max(args.steps * .75, 1)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, detail = model(base, transported, support, cleanup, cleanup_support,
                                       action, pose, arm, labels, teacher_forcing,
                                       args.mask_threshold, args.active_from, enabled_layers)
            loss, parts = loss_terms(prediction, truth, base, labels, detail)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0))
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if step == 1 or step % 25 == 0:
            record = {"step": step, "loss": float(loss), "teacher_forcing": teacher_forcing,
                      "gradient": gradient, "lr": scheduler.get_last_lr()[0],
                      "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
            record.update({key: float(value) for key, value in parts.items()})
            print(json.dumps(record), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            model.eval(); metrics = evaluate(model, dev, arrays, geometry, actions, arms, stats,
                                             device, args.validation_windows, args.workers,
                                             args.mask_threshold, args.active_from,
                                             enabled_layers); model.train()
            history.append({"step": step, "metrics": metrics}); print(json.dumps(history[-1]), flush=True)
            payload = {"format": "track2-layered-object-state-parent-v24.0", "step": step,
                       "state_dict": model.state_dict(), "channels": args.channels,
                       "maximum_residual_flow": model.maximum_residual_flow,
                       "action_mean": mean, "action_std": std, "metrics": metrics}
            atomic_save(payload, output / "latest.pt")
            if metrics["gate_score"] > best:
                best = metrics["gate_score"]; atomic_save(payload, output / "best.pt")
            atomic_save({"format": "track2-v24-training-state", "step": step,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "best": best, "history": history, "rng_state": rng.bit_generator.state},
                        state_path)
    executor.shutdown()
    manifest = {"format": "track2-layered-object-state-parent-v24.0-training",
                "data_boundary": "supplied_50_episodes_only", "train_windows": len(train),
                "dev_windows": len(dev), "train_arm_counts": {str(k): len(v) for k, v in by_arm.items()},
                "positive_layer_arm_counts": {str(k): len(v) for k, v in positive_by_arm.items()},
                "steps": args.steps, "batch_size": args.batch_size,
                "parameter_count": parameter_count(args.channels), "best_gate_score": best,
                "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
