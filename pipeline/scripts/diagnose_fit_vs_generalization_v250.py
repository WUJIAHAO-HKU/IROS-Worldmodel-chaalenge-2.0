#!/usr/bin/env python3
"""Controlled v25 diagnostic: representation oracle and 32-window overfit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from wam_pipeline.direct_overfit_diagnostic_v250 import DirectOverfitDiagnosticV250, parameter_count
from wam_pipeline.geometry_visibility_router_v172 import full_layer_masks
from train_layered_object_state_parent_v240 import episode, load_actions, make_example


REGION = (72, 232, 30, 210)


def balanced(indices, arms, per_arm):
    result = []
    for arm in (0, 1):
        values = [index for index in indices if arms[index] == arm]
        positions = np.linspace(0, len(values) - 1, per_arm, dtype=int)
        result.extend(values[position] for position in positions)
    return result


def build_labels(target, arms, indices):
    output = []
    for order, index in enumerate(indices):
        side = "left" if int(arms[index]) == 0 else "right"
        output.append(np.stack([full_layer_masks(frame, side)[:4] for frame in target[index]]))
        if (order + 1) % 16 == 0:
            print(json.dumps({"labels": order + 1, "total": len(indices)}), flush=True)
    return np.stack(output).astype(np.uint8)


def tensor_batch(parent, target, context, actions, arms, labels, positions, mean, std, device):
    indices = np.asarray(positions, np.int64)
    rgb = lambda value: torch.from_numpy(value[indices].transpose(0, 1, 4, 2, 3).copy()).to(device).float() / 255
    base = rgb(parent); truth = rgb(target)
    observed = torch.from_numpy(context[indices].transpose(0, 1, 4, 2, 3).copy()).to(device).float() / 255
    normalized = np.stack([(actions[i] - mean[arms[i]]) / std[arms[i]] for i in indices])
    action = torch.from_numpy(normalized).to(device).float()
    arm = torch.from_numpy(arms[indices]).to(device).long()
    layer = torch.from_numpy(labels).to(device).float()
    return base, truth, observed, action, arm, layer


def highpass(value):
    flat = value.flatten(0, 1)
    return (flat - F.avg_pool2d(flat, 5, 1, 2, count_include_pad=False)).reshape_as(value)


def objective(output, base, truth, context, labels):
    objects = labels.max(2, keepdim=True).values
    previous_truth = torch.cat((context[:, -1:], truth[:, :-1]), 1)
    motion = F.max_pool2d(((truth - previous_truth).abs().mean(2, keepdim=True) > .012)
                          .flatten(0, 1).float(), 7, 1, 3).reshape_as(objects)
    weights = 1 + 5 * objects + 2 * motion
    pixel = ((output - truth).abs() * weights).sum() / (weights.sum() * 3)
    texture = ((highpass(output) - highpass(truth)).abs() * (1 + 4 * objects)).sum()
    texture /= ((1 + 4 * objects).sum() * 3)
    target_delta = torch.cat((context[:, -1:], truth), 1).diff(dim=1)
    temporal = (torch.cat((context[:, -1:], output), 1).diff(dim=1) - target_delta).abs().mean()
    return pixel + .35 * texture + .20 * temporal, {
        "pixel": pixel, "texture": texture, "temporal": temporal,
    }


def metric_parts(prediction, truth, context, labels):
    y0, y1, x0, x1 = REGION
    values = {
        "rgb": (prediction - truth).abs().mean((2, 3, 4)),
        "texture": (highpass(prediction) - highpass(truth)).abs().mean((2, 3, 4)),
        "temporal": (torch.cat((context[:, -1:], prediction), 1).diff(dim=1) -
                     torch.cat((context[:, -1:], truth), 1).diff(dim=1)).abs().mean((2, 3, 4)),
        "contact": (prediction[:, :, :, y0:y1, x0:x1] -
                    truth[:, :, :, y0:y1, x0:x1]).abs().mean((2, 3, 4)),
    }
    structure = labels[:, :, 1:3].max(2).values[:, :, None].expand(-1, -1, 3, -1, -1)
    values["structure"] = ((prediction - truth).abs() * structure).sum((2, 3, 4))
    values["structure"] /= structure.sum((2, 3, 4)).clamp_min(1)
    return values


@torch.inference_mode()
def evaluate_model(model, parent, target, context, actions, arms, labels, indices,
                   mean, std, device):
    names = ("rgb", "texture", "temporal", "contact", "structure")
    before = {name: torch.zeros(8) for name in names}; after = {name: torch.zeros(8) for name in names}
    for local, index in enumerate(indices):
        values = tensor_batch(parent, target, context, actions, arms, labels[local:local + 1],
                              [index], mean, std, device)
        base, truth, observed, action, arm, layer = values
        output = model(base, observed, action, arm)
        for name, value in metric_parts(base, truth, observed, layer).items(): before[name] += value[0].cpu()
        for name, value in metric_parts(output, truth, observed, layer).items(): after[name] += value[0].cpu()
    result = {"windows": len(indices)}
    gains = []
    for name in names:
        p = 255 * before[name] / len(indices); v = 255 * after[name] / len(indices)
        gain = 100 * (p - v) / p.clamp_min(1e-9)
        result[f"parent_{name}_mae"] = float(p.mean()); result[f"prediction_{name}_mae"] = float(v.mean())
        result[f"{name}_improvement_percent"] = float(gain.mean())
        result[f"frame_{name}_improvement_percent"] = gain.tolist(); gains.append(float(gain.min()))
    result["minimum_horizon_gain_percent"] = min(gains)
    return result


def oracle_sequence(index, parent, target, context, geometry, arms):
    example = make_example(index, parent, target, context, geometry[0], geometry[1], arms)
    base, truth, transported, support, cleanup, cleanup_support, labels = example
    output = base.copy(); best = np.abs(base.astype(np.float32) - truth).mean(-1)
    for time in range(8):
        for layer in range(4):
            valid = support[time, layer].astype(bool)
            error = np.abs(transported[time].astype(np.float32) - truth[time]).mean(-1)
            use = valid & (error < best[time]); output[time][use] = transported[time][use]
            best[time][use] = error[use]
        valid = cleanup_support[time, 0].astype(bool)
        error = np.abs(cleanup[time].astype(np.float32) - truth[time]).mean(-1)
        use = valid & (error < best[time]); output[time][use] = cleanup[time][use]
        best[time][use] = error[use]
    return output, labels


@torch.inference_mode()
def evaluate_oracle(indices, parent, target, context, geometry, arms, device):
    names = ("rgb", "texture", "temporal", "contact", "structure")
    before = {name: torch.zeros(8) for name in names}; after = {name: torch.zeros(8) for name in names}
    for order, index in enumerate(indices):
        oracle, labels = oracle_sequence(index, parent, target, context, geometry, arms)
        def rgb(value): return torch.from_numpy(value.transpose(0, 3, 1, 2).copy())[None].to(device).float() / 255
        base = rgb(parent[index]); truth = rgb(target[index]); output = rgb(oracle)
        observed = torch.from_numpy(context[index].transpose(0, 3, 1, 2).copy())[None].to(device).float() / 255
        layer = torch.from_numpy(labels)[None].to(device).float()
        for name, value in metric_parts(base, truth, observed, layer).items(): before[name] += value[0].cpu()
        for name, value in metric_parts(output, truth, observed, layer).items(): after[name] += value[0].cpu()
    result = {"windows": len(indices), "oracle_uses_future_truth": True}
    for name in names:
        p = 255 * before[name] / len(indices); v = 255 * after[name] / len(indices)
        result[f"parent_{name}_mae"] = float(p.mean()); result[f"oracle_{name}_mae"] = float(v.mean())
        result[f"{name}_upper_bound_improvement_percent"] = float(100 * (p.mean() - v.mean()) / p.mean())
    return result


def atomic_save(value, path):
    temporary = path.with_suffix(f".tmp.{os.getpid()}"); torch.save(value, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--precompute-cache", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(1)
    device = torch.device("cuda"); windows = Path(args.windows); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; target = cache["target"]; context = cache["context"]; names = cache["windows"].astype(str)
    with np.load(args.precompute_cache, allow_pickle=False) as cache:
        if not np.array_equal(cache["windows"].astype(str), names): raise ValueError("cache identity mismatch")
        source = cache["source_geometry"]; future = cache["future_geometry"]
    actions, arms = load_actions(windows, names)
    development = [i for i, name in enumerate(names) if episode(name) in (36, 47)]
    training = [i for i, name in enumerate(names) if episode(name) not in (36, 47)]
    train_indices = balanced(training, arms, 16); dev_indices = balanced(development, arms, 16)
    selected = train_indices + dev_indices; selected_labels = build_labels(target, arms, selected)
    train_labels = selected_labels[:32]; dev_labels = selected_labels[32:]
    mean = np.stack([actions[train_indices][arms[train_indices] == arm].mean((0, 1)) for arm in (0, 1)])
    std = np.stack([actions[train_indices][arms[train_indices] == arm].std((0, 1)) for arm in (0, 1)])
    std = np.maximum(std, 1e-4)
    selection = {"format": "track2-v25-fit-diagnostic-selection", "train_windows": names[train_indices].tolist(),
                 "dev_windows": names[dev_indices].tolist(), "train_arm_counts": np.bincount(arms[train_indices], minlength=2).tolist(),
                 "dev_arm_counts": np.bincount(arms[dev_indices], minlength=2).tolist()}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    oracle = {"train32": evaluate_oracle(train_indices, parent, target, context, (source, future), arms, device),
              "dev32": evaluate_oracle(dev_indices, parent, target, context, (source, future), arms, device)}
    (output / "representation_oracle.json").write_text(json.dumps(oracle, indent=2) + "\n")
    print(json.dumps({"oracle": oracle}), flush=True)
    model = DirectOverfitDiagnosticV250(args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(args.seed); history = []
    milestones = {0, 100, 250, 500, 1000, args.steps}
    for step in range(args.steps + 1):
        if step in milestones:
            model.eval(); train_metrics = evaluate_model(model, parent, target, context, actions, arms, train_labels,
                                                         train_indices, mean, std, device)
            dev_metrics = evaluate_model(model, parent, target, context, actions, arms, dev_labels,
                                                       dev_indices, mean, std, device); model.train()
            record = {"step": step, "train32": train_metrics, "dev32": dev_metrics}; history.append(record)
            print(json.dumps(record), flush=True)
            atomic_save({"format": "track2-direct-overfit-diagnostic-v25.0", "step": step,
                         "channels": args.channels, "state_dict": model.state_dict(),
                         "action_mean": mean, "action_std": std, "train": train_metrics,
                         "dev": dev_metrics}, output / f"step_{step:04d}.pt")
        if step == args.steps: break
        local = rng.choice(32, args.batch_size, replace=True).tolist()
        batch_indices = [train_indices[value] for value in local]
        labels = train_labels[local]
        base, truth, observed, action, arm, layer = tensor_batch(
            parent, target, context, actions, arms, labels, batch_indices, mean, std, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(base, observed, action, arm)
            loss, parts = objective(prediction, base, truth, observed, layer)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2)); scaler.step(optimizer); scaler.update(); scheduler.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss), "gradient": gradient,
                              "lr": scheduler.get_last_lr()[0], "peak_memory_gib": torch.cuda.max_memory_allocated()/2**30,
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
    report = {"format": "track2-v25-fit-vs-generalization-diagnostic", "data_boundary": "supplied_50_episodes_only",
              "parameters": parameter_count(args.channels), "steps": args.steps, "oracle": oracle,
              "history": history}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
