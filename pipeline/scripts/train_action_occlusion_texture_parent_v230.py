#!/usr/bin/env python3
"""Train the recurrent v23 structure/texture parent extension."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from wam_pipeline.action_occlusion_texture_parent_v230 import (
    ActionOcclusionTextureParentV230,
    parameter_count,
)
from wam_pipeline.autoregressive_unet import OneStepActionUNet
from train_autoregressive_texture_memory_v190 import highpass, window_arm_labels
from train_autoregressive_unet import WindowDataset, frames_for_model, window_motion_scores
from train_trajectory_motion_renderer_v220 import load_pose, project_future


REGION = (72, 232, 30, 210)


def load_rgb_parent(path: Path, device: torch.device):
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError("v23 requires a track2 autoregressive RGB parent")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    normalization = np.load(path / "action_normalization.npz")
    mean = torch.from_numpy(normalization["mean"]).to(device)
    std = torch.from_numpy(normalization["std"]).to(device)
    return model, mean, std


def rgb_parent_rollout(parent, context, history, future):
    values = []
    for action in future.unbind(1):
        value = parent(context, torch.cat((history, action[:, None]), 1)).clamp(0, 1)
        values.append(value)
        context = torch.cat((context[:, 1:], value[:, None]), 1)
        history = torch.cat((history[:, 1:], action[:, None]), 1)
    return torch.stack(values, 1)


def recurrent_rollout(
    parent, extension, context, history, future, memory, poses, arms,
    state_strength=1.0, output_strengths=None, residual_ema=1.0,
):
    """Roll out v23 and feed each corrected frame back into the frozen parent.

    The recurrent state is detached between steps.  Training therefore observes
    the deployed state distribution without retaining eight copies of the
    frozen 8M-parameter parent graph.
    """
    predictions, bases, diagnostics = [], [], []
    previous_residual = None
    for index, action in enumerate(future.unbind(1)):
        with torch.no_grad():
            base = parent(context.detach(), torch.cat((history, action[:, None]), 1)).clamp(0, 1)
        horizon = base.new_full((len(base), 1), (index + 1) / future.shape[1])
        value, detail = extension(
            base.float(), context[:, -1].detach().float(), memory.float(), action.float(),
            poses[:, index].float(), horizon, arms,
        )
        residual = value - base
        if previous_residual is not None and residual_ema < 1.0:
            residual = float(residual_ema) * residual + (1 - float(residual_ema)) * previous_residual
            value = (base + residual).clamp(0, 1)
        previous_residual = residual
        if output_strengths is not None:
            strength = float(output_strengths[index])
            value = base + strength * (value - base)
        predictions.append(value)
        bases.append(base)
        diagnostics.append(detail)
        state = base + float(state_strength) * (value.detach() - base)
        context = torch.cat((context[:, 1:].detach(), state[:, None]), 1)
        history = torch.cat((history[:, 1:], action[:, None]), 1)
    return torch.stack(predictions, 1), torch.stack(bases, 1), diagnostics


def metric_tensors(value, target, context_last):
    y0, y1, x0, x1 = REGION
    rgb = (value - target).abs().mean((2, 3, 4))
    texture = (highpass(value.flatten(0, 1)) - highpass(target.flatten(0, 1))).abs()
    texture = texture.mean((1, 2, 3)).reshape(len(value), value.shape[1])
    target_delta = torch.cat((context_last[:, None], target), 1).diff(dim=1)
    temporal = (torch.cat((context_last[:, None], value), 1).diff(dim=1) - target_delta).abs().mean((2, 3, 4))
    contact = (value[:, :, :, y0:y1, x0:x1] - target[:, :, :, y0:y1, x0:x1]).abs().mean((2, 3, 4))
    return rgb, texture, temporal, contact


def objective(
    output, bases, target, context_last, diagnostics,
    temporal_weight=.50, texture_weight=.20, contact_weight=.30, protect_weight=1.0,
):
    previous_target = torch.cat((context_last[:, None], target[:, :-1]), 1)
    motion = (target - previous_target).abs().mean(2, keepdim=True)
    motion_support = F.max_pool2d((motion > .012).flatten(0, 1).float(), 9, 1, 4).reshape_as(motion)
    target_high = highpass(target.flatten(0, 1)).reshape_as(target)
    edge_support = F.max_pool2d(
        (target_high.abs().mean(2, keepdim=True) > .012).flatten(0, 1).float(), 5, 1, 2
    ).reshape_as(motion)
    dark = (target.mean(2, keepdim=True) < .32).float()
    green = ((target[:, :, 1:2] > 1.15 * target[:, :, 0:1]) &
             (target[:, :, 1:2] > 1.08 * target[:, :, 2:3])).float()
    roi = torch.zeros_like(motion)
    y0, y1, x0, x1 = REGION
    roi[:, :, :, y0:y1, x0:x1] = 1
    object_support = roi * torch.maximum(edge_support, torch.maximum(dark, green))
    weights = 1 + 1.5 * motion_support + .75 * edge_support + .5 * object_support
    pixel = ((output - target).abs() * weights).sum() / (weights.sum() * 3)
    texture = ((highpass(output.flatten(0, 1)).reshape_as(output) - target_high).abs()
               * (1 + 1.5 * edge_support)).sum() / ((1 + 1.5 * edge_support).sum() * 3)
    target_delta = torch.cat((context_last[:, None], target), 1).diff(dim=1)
    temporal = (torch.cat((context_last[:, None], output), 1).diff(dim=1) - target_delta).abs().mean()
    contact = (output[:, :, :, y0:y1, x0:x1] - target[:, :, :, y0:y1, x0:x1]).abs().mean()
    accurate = (bases - target).abs().mean(2, keepdim=True) < (2 / 255)
    protect = (output - bases).abs()[accurate.expand_as(output)].mean() if accurate.any() else output.new_zeros(())
    flows = torch.stack([item["flow"] for item in diagnostics], 1)
    flow_smooth = (flows[..., 1:, :] - flows[..., :-1, :]).abs().mean()
    flow_smooth = flow_smooth + (flows[..., 1:] - flows[..., :-1]).abs().mean()
    flow_magnitude = flows.abs().mean()
    structure_magnitude = torch.stack([item["structure_residual"].abs().mean() for item in diagnostics]).mean()
    texture_magnitude = torch.stack([item["texture_residual"].abs().mean() for item in diagnostics]).mean()
    loss = (pixel + temporal_weight * temporal + texture_weight * texture
            + contact_weight * contact + protect_weight * protect
            + .005 * flow_smooth + .001 * flow_magnitude
            + .01 * structure_magnitude + .01 * texture_magnitude)
    return loss, {
        "pixel": pixel, "texture": texture, "temporal": temporal, "contact": contact,
        "protect": protect, "flow": flow_magnitude, "structure": structure_magnitude,
        "texture_residual": texture_magnitude,
    }


def source_arm_sampling_weights(
    dataset: WindowDataset,
    motion: np.ndarray,
    arm_labels: np.ndarray,
    source_manifest: Path,
    synthetic_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Balance official/synthetic domains and arms without changing the dataset.

    Motion oversampling is normalized independently inside each source/arm
    stratum so a high-motion-heavy stratum cannot silently change the requested
    domain mixture.
    """
    if not 0.0 < synthetic_fraction < 1.0:
        raise ValueError("synthetic sampling fraction must be in (0, 1)")
    records = json.loads(source_manifest.read_text())
    if not isinstance(records, list):
        raise ValueError("source manifest must be a JSON list")
    by_path = {str(record["path"]): str(record["source"]) for record in records}
    sources = np.asarray([by_path.get(path.name, "") for path in dataset.paths])
    if not np.isin(sources, ("official", "synthetic")).all():
        missing = [dataset.paths[index].name for index in np.flatnonzero(~np.isin(sources, ("official", "synthetic")))[:5]]
        raise ValueError(f"source manifest does not cover training windows: {missing}")
    motion_weights = 1.0 + 4.0 * (motion >= 0.03)
    weights = np.zeros(len(dataset), dtype=np.float64)
    counts = {}
    expected = {}
    for source, source_mass in (("official", 1.0 - synthetic_fraction), ("synthetic", synthetic_fraction)):
        for arm in (0, 1):
            mask = (sources == source) & (arm_labels == arm)
            count = int(mask.sum())
            if count == 0:
                raise ValueError(f"empty sampling stratum: {source}/arm{arm}")
            stratum_mass = source_mass / 2.0
            weights[mask] = stratum_mass * motion_weights[mask] / motion_weights[mask].sum()
            key = f"{source}_arm{arm}"
            counts[key] = count
            expected[key] = stratum_mass
    weights *= len(weights) / weights.sum()
    return weights, {
        "mode": "source_arm_balanced",
        "source_manifest": str(source_manifest.resolve()),
        "synthetic_fraction": synthetic_fraction,
        "motion_threshold": 0.03,
        "high_motion_multiplier": 5.0,
        "counts": counts,
        "expected_sampling_fraction": expected,
    }


@torch.inference_mode()
def evaluate(
    loader, parent, pose_model, pose_stats, extension, device, mean, std,
    state_strength=1.0, output_strengths=None, residual_ema=1.0,
):
    extension.eval()
    parent_sums = [torch.zeros(8) for _ in range(4)]
    value_sums = [torch.zeros(8) for _ in range(4)]
    arms = {0: [[0., 0.] for _ in range(4)] + [0], 1: [[0., 0.] for _ in range(4)] + [0]}
    diagnostics_sum = np.zeros(4, dtype=np.float64)
    count = 0
    for raw_context, raw_history, raw_future, raw_target in loader:
        raw_history = raw_history.to(device)
        raw_future = raw_future.to(device)
        context = frames_for_model(raw_context).to(device)
        target = frames_for_model(raw_target).to(device)
        poses, active_arms = project_future(pose_model, pose_stats, raw_history, raw_future)
        history = ((raw_history - mean) / std).float()
        future = ((raw_future - mean) / std).float()
        parent_value = rgb_parent_rollout(parent, context.clone(), history.clone(), future)
        value, _, details = recurrent_rollout(
            parent, extension, context, history, future, context.clone(), poses, active_arms,
            state_strength, output_strengths, residual_ema,
        )
        parent_metrics = metric_tensors(parent_value, target, context[:, -1])
        value_metrics = metric_tensors(value, target, context[:, -1])
        for index, (before, after) in enumerate(zip(parent_metrics, value_metrics)):
            parent_sums[index] += before.sum(0).cpu()
            value_sums[index] += after.sum(0).cpu()
        for row, arm in enumerate(active_arms.tolist()):
            for index, (before, after) in enumerate(zip(parent_metrics, value_metrics)):
                arms[arm][index][0] += float(before[row].mean())
                arms[arm][index][1] += float(after[row].mean())
            arms[arm][4] += 1
        diagnostics_sum += np.asarray([
            np.mean([float(item["flow"].abs().mean()) for item in details]),
            np.mean([float(item["flow_gate"].mean()) for item in details]),
            np.mean([float(item["structure_gate"].mean()) for item in details]),
            np.mean([float(item["texture_gate"].mean()) for item in details]),
        ]) * len(context)
        count += len(context)
    names = ("rgb", "texture", "temporal", "contact")
    result = {"windows": count, "arms": {}}
    scores = []
    for index, name in enumerate(names):
        before = 255 * parent_sums[index] / count
        after = 255 * value_sums[index] / count
        frame_gain = 100 * (before - after) / before
        result[f"parent_{name}_mae"] = float(before.mean())
        result[f"{name}_mae"] = float(after.mean())
        result[f"{name}_improvement_percent"] = float(100 * (before.mean() - after.mean()) / before.mean())
        result[f"frame_{name}_improvement_percent"] = frame_gain.tolist()
        scores.extend((result[f"{name}_improvement_percent"], float(frame_gain.min())))
    for arm, record in arms.items():
        arm_result = {"windows": int(record[4])}
        for index, name in enumerate(names):
            before, after = record[index]
            arm_result[f"{name}_improvement_percent"] = 100 * (before - after) / before
            scores.append(arm_result[f"{name}_improvement_percent"])
        result["arms"][f"arm{arm}"] = arm_result
    diagnostics_sum /= count
    result.update({
        "mean_flow_px": float(diagnostics_sum[0]),
        "mean_flow_gate": float(diagnostics_sum[1]),
        "mean_structure_gate": float(diagnostics_sum[2]),
        "mean_texture_gate": float(diagnostics_sum[3]),
        "gate_score": min(scores),
    })
    return result


def save(path: Path, model, metadata):
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "parent_extension.pt.tmp"
    torch.save({
        "format": "track2-action-occlusion-texture-parent-v23.0",
        "state_dict": model.state_dict(),
        "config": {
            "maximum_flow_pixels": model.maximum_flow_pixels,
            "maximum_structure_residual": model.maximum_structure_residual,
            "maximum_texture_residual": model.maximum_texture_residual,
            "initial_gate_logit": model.initial_gate_logit,
        },
    }, temporary)
    os.replace(temporary, path / "parent_extension.pt")
    (path / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-windows", type=int, default=16)
    parser.add_argument("--state-strength", type=float, default=0.0)
    parser.add_argument("--temporal-weight", type=float, default=.50)
    parser.add_argument("--texture-weight", type=float, default=.20)
    parser.add_argument("--contact-weight", type=float, default=.30)
    parser.add_argument("--protect-weight", type=float, default=1.0)
    parser.add_argument("--maximum-flow-pixels", type=float, default=6.0)
    parser.add_argument("--maximum-structure-residual", type=float, default=.10)
    parser.add_argument("--maximum-texture-residual", type=float, default=.05)
    parser.add_argument("--initial-gate-logit", type=float, default=-4.0)
    parser.add_argument("--init-extension")
    parser.add_argument("--data-boundary", default="supplied_50_episodes_only")
    parser.add_argument("--evaluation-residual-ema", type=float, default=1.0)
    parser.add_argument("--source-manifest")
    parser.add_argument("--synthetic-sampling-fraction", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    if not 0.0 < args.evaluation_residual_ema <= 1.0:
        raise ValueError("evaluation-residual-ema must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    split = json.loads(Path(args.split_manifest).read_text())
    train = WindowDataset(Path(args.windows), split["train_episodes"])
    dev = WindowDataset(Path(args.windows), split["validation_episodes"])
    parent, mean, std = load_rgb_parent(Path(args.init_checkpoint), device)
    pose_model, pose_stats = load_pose(args.pose_checkpoint, device)
    motion = window_motion_scores(train)
    arm_labels = window_arm_labels(train)
    if args.source_manifest:
        weights, sampling = source_arm_sampling_weights(
            train, motion, arm_labels, Path(args.source_manifest), args.synthetic_sampling_fraction
        )
    else:
        arm_counts = np.bincount(arm_labels, minlength=2)
        weights = (1 + 4 * (motion >= .03)) * (len(arm_labels) / (2 * arm_counts[arm_labels]))
        sampling = {
            "mode": "arm_balanced",
            "motion_threshold": .03,
            "high_motion_multiplier": 5.0,
            "arm_counts": arm_counts.tolist(),
        }
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    positions = np.linspace(0, len(dev) - 1, min(len(dev), args.validation_windows), dtype=int).tolist()
    dev_loader = DataLoader(Subset(dev, positions), batch_size=1, num_workers=2, pin_memory=True)
    model = ActionOcclusionTextureParentV230(
        maximum_flow_pixels=args.maximum_flow_pixels,
        maximum_structure_residual=args.maximum_structure_residual,
        maximum_texture_residual=args.maximum_texture_residual,
        initial_gate_logit=args.initial_gate_logit,
    ).to(device)
    if args.init_extension:
        initialization = torch.load(
            Path(args.init_extension) / "parent_extension.pt", map_location="cpu", weights_only=True
        )
        if initialization.get("format") != "track2-action-occlusion-texture-parent-v23.0":
            raise ValueError("unsupported v23 initialization checkpoint")
        model.load_state_dict(initialization["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    iterator = iter(loader)
    output = Path(args.output)
    history_log = []
    best = -1e9
    for step in range(1, args.steps + 1):
        try:
            raw_context, raw_history, raw_future, raw_target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_context, raw_history, raw_future, raw_target = next(iterator)
        raw_history = raw_history.to(device)
        raw_future = raw_future.to(device)
        context = frames_for_model(raw_context).to(device)
        target = frames_for_model(raw_target).to(device)
        poses, active_arms = project_future(pose_model, pose_stats, raw_history, raw_future)
        history = ((raw_history - mean) / std).float()
        future = ((raw_future - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, bases, details = recurrent_rollout(
                parent, model, context, history, future, context.clone(), poses, active_arms,
                state_strength=args.state_strength,
            )
            loss, parts = objective(
                prediction, bases, target, context[:, -1], details,
                temporal_weight=args.temporal_weight,
                texture_weight=args.texture_weight,
                contact_weight=args.contact_weight,
                protect_weight=args.protect_weight,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite v23 loss at step {step}")
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(json.dumps({
                "step": step, "loss": float(loss), **{key: float(value) for key, value in parts.items()},
                "gradient": gradient, "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(
                dev_loader, parent, pose_model, pose_stats, model, device, mean, std,
                state_strength=args.state_strength,
                residual_ema=args.evaluation_residual_ema,
            )
            model.train()
            record = {"step": step, "metrics": metrics}
            history_log.append(record)
            print(json.dumps(record), flush=True)
            metadata = {
                "format": "track2-v23.1-stable-state-parent-training",
                "data_boundary": args.data_boundary,
                "step": step,
                "parameters": parameter_count(),
                "frozen_rgb_parent": True,
                "rgb_state_writeback": args.state_strength > 0,
                "state_strength": args.state_strength,
                "evaluation_residual_ema": args.evaluation_residual_ema,
                "sampling": sampling,
                "loss_weights": {
                    "temporal": args.temporal_weight,
                    "texture": args.texture_weight,
                    "contact": args.contact_weight,
                    "protect": args.protect_weight,
                },
                "model_config": {
                    "maximum_flow_pixels": args.maximum_flow_pixels,
                    "maximum_structure_residual": args.maximum_structure_residual,
                    "maximum_texture_residual": args.maximum_texture_residual,
                    "initial_gate_logit": args.initial_gate_logit,
                },
                "initialization_checkpoint": args.init_extension,
                "spatial_pose_heatmaps": True,
                "separate_flow_structure_texture_paths": True,
                "train_windows": len(train),
                "dev_windows": len(dev),
                "history": history_log,
                "metrics": metrics,
            }
            save(output / f"step_{step:04d}", model, metadata)
            save(output / "latest", model, metadata)
            if metrics["gate_score"] > best:
                best = metrics["gate_score"]
                save(output / "best", model, metadata)


if __name__ == "__main__":
    main()
