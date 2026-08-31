#!/usr/bin/env python3
"""Evaluate structure-locked multiband texture reprojection for v9.2."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet
from wam_pipeline.protected_layered_flow_v91 import ProtectedLayeredFlowV91


def load_model(base_checkpoint: Path, head_checkpoint: Path, device: torch.device):
    with np.load(base_checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        base = MultiSourceActionFlowUNet(int(config["base_channels"]))
    state = torch.load(base_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    base.load_state_dict(state["state_dict"], strict=True)
    head = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    model = ProtectedLayeredFlowV91(base, int(head["base_channels"]), float(head["max_residual_flow"]))
    incompatible = model.load_state_dict(head["state_dict"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("base_flow_model.")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"invalid flow head: {missing} {incompatible.unexpected_keys}")
    return model.to(device).eval(), head["active_mean"].to(device), head["active_std"].to(device)


def frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)


def blur(value: torch.Tensor, kernel: int) -> torch.Tensor:
    shape = value.shape
    flat = value.flatten(0, -4)
    result = F.avg_pool2d(flat, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return result.reshape(shape)


def block_pool(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    pooled = F.avg_pool2d(value.flatten(0, -3)[:, None], 4, stride=4).squeeze(1)
    return pooled.reshape(*shape[:-2], *pooled.shape[-2:])


def expand_choice(choice: torch.Tensor) -> torch.Tensor:
    return choice.repeat_interleave(4, -2).repeat_interleave(4, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--selection", choices=("uniform", "top-motion"), default="uniform")
    parser.add_argument("--fixed-kernel", type=int)
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fixed = (args.fixed_kernel, args.fixed_alpha, args.fixed_threshold)
    if any(value is not None for value in fixed) and not all(value is not None for value in fixed):
        raise ValueError("fixed-kernel, fixed-alpha, and fixed-threshold must be supplied together")

    device = torch.device(args.device)
    base_checkpoint = Path(args.base_checkpoint)
    model, active_mean, active_std = load_model(base_checkpoint, Path(args.flow_head), device)
    with np.load(base_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent_cache, names = cache["prediction"], cache["windows"].astype(str).tolist()

    motion = []
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            previous = np.concatenate((window["context_frames"][-1:].astype(np.float32), target[:-1]), axis=0)
        motion.append((float(np.abs(target - previous).mean() / 255.0), index))
    if args.selection == "top-motion":
        selected = sorted(motion, reverse=True)[: args.samples]
    else:
        indices = np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=np.int64)
        scores = {index: score for score, index in motion}
        selected = [(scores[int(index)], int(index)) for index in indices]

    if all(value is not None for value in fixed):
        configurations = [fixed]
    else:
        configurations = list(itertools.product((5, 9, 17), (0.25, 0.5, 0.75, 1.0), (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)))
    sums = {configuration: 0.0 for configuration in configurations}
    oracle_sums = {configuration: 0.0 for configuration in configurations}
    selected_blocks = {configuration: 0 for configuration in configurations}
    baseline_sum = moving_baseline = 0.0
    moving_sums = {configuration: 0.0 for configuration in configurations}
    total = moving_count = block_count = 0

    with torch.inference_mode():
        for _, index in selected:
            name = names[index]
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                context = frames(window["context_frames"], device)[None]
                target = frames(window["target_frames"], device)[None]
                actions_np = np.concatenate((window["history_actions"], window["future_actions"]), axis=0)
            parent = frames(parent_cache[index], device)[None]
            actions = torch.from_numpy(actions_np.copy()).to(device).float()[None]
            active, arm = model.active_arm_actions(actions)
            result = model(context, (actions - action_mean) / action_std,
                           (active - active_mean) / active_std, arm, parent)
            warps = result["candidates"][:, :, 1:]
            parent_sources = parent[:, :, None].expand_as(warps)
            previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
            moving = (target - previous).abs().mean(dim=2) >= 0.03
            parent_error = (parent - target).abs()
            baseline_sum += float(parent_error.sum())
            moving_baseline += float((parent_error * moving[:, :, None]).sum())

            for kernel in sorted({int(configuration[0]) for configuration in configurations}):
                parent_low = blur(parent_sources, kernel)
                warp_low = blur(warps, kernel)
                high_delta = (warps - warp_low) - (parent_sources - parent_low)
                low_disagreement = block_pool((warp_low - parent_low).abs().mean(dim=3))
                source = low_disagreement.argmin(dim=2)
                best_low = low_disagreement.gather(2, source[:, :, None]).squeeze(2)
                full_source = expand_choice(source)
                for configuration in [item for item in configurations if int(item[0]) == kernel]:
                    _, alpha, threshold = configuration
                    transported = (parent_sources + float(alpha) * high_delta).clamp(0, 1)
                    chosen = transported.gather(
                        2, full_source[:, :, None, None].expand(-1, -1, 1, 3, 256, 256)
                    ).squeeze(2)
                    gate = expand_choice(best_low * 255.0 < float(threshold))
                    prediction = torch.where(gate[:, :, None], chosen, parent)
                    error = (prediction - target).abs()
                    sums[configuration] += float(error.sum())
                    moving_sums[configuration] += float((error * moving[:, :, None]).sum())
                    selected_blocks[configuration] += int(gate.sum() // 16)
                    candidate_error = block_pool((transported - target[:, :, None]).abs().mean(dim=3))
                    parent_block_error = block_pool(parent_error.mean(dim=2))[:, :, None]
                    oracle_error = torch.cat((parent_block_error, candidate_error), dim=2)
                    oracle_sums[configuration] += float(oracle_error.min(dim=2).values.sum()) * 3 * 16
            total += target.numel()
            moving_count += int(moving.sum()) * 3
            block_count += target.shape[0] * target.shape[1] * 64 * 64

    baseline = 255 * baseline_sum / total
    rows = []
    for configuration in configurations:
        rgb = 255 * sums[configuration] / total
        oracle = 255 * oracle_sums[configuration] / total
        rows.append({
            "kernel": int(configuration[0]), "alpha": float(configuration[1]),
            "lowpass_threshold_255": float(configuration[2]), "rgb_mae": rgb,
            "improvement_percent": 100 * (baseline - rgb) / baseline,
            "block4_oracle_rgb_mae": oracle,
            "block4_oracle_improvement_percent": 100 * (baseline - oracle) / baseline,
            "moving_rgb_mae": 255 * moving_sums[configuration] / moving_count,
            "selected_block_fraction": selected_blocks[configuration] / block_count,
        })
    best = min(rows, key=lambda row: row["rgb_mae"])
    result = {
        "format": "track2-protected-multiband-transport-v9.2-eval",
        "sample_count": len(selected), "selection": args.selection,
        "baseline_rgb_mae": baseline, "moving_baseline_rgb_mae": 255 * moving_baseline / moving_count,
        "best": best, "grid": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
