#!/usr/bin/env python3
"""Calibrate a safe renderer for the eight-frame trajectory selector.

The selector chooses one coherent native-resolution observation per 8x8 patch.
Only its high-frequency residual is injected into the frozen parent prediction;
the parent trajectory is never fed the rendered frame, preventing error feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet
from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector
from train_autoregressive_texture_memory_v190 import highpass
from train_autoregressive_unet import WindowDataset, frames_for_model


def load_parent(init_checkpoint: Path, memory_checkpoint: Path, device: torch.device):
    parent = OneStepActionTextureMemoryUNet().to(device)
    initial = torch.load(init_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = parent.load_state_dict(initial["state_dict"], strict=False)
    allowed = ("memory_enc0", "memory_fusion", "memory_head", "benefit_head", "benefit_expert", "source_expert")
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise ValueError((missing, unexpected))
    parent.initialize_memory_encoder()
    memory = torch.load(memory_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = parent.load_state_dict(memory["state_dict"], strict=False)
    if unexpected or any(not key.startswith(("benefit_head", "benefit_expert", "source_expert")) for key in missing):
        raise ValueError((missing, unexpected))
    parent.eval().requires_grad_(False)
    normalization = np.load(init_checkpoint / "action_normalization.npz")
    return parent, torch.from_numpy(normalization["mean"]).to(device), torch.from_numpy(normalization["std"]).to(device)


def arm_index(history: torch.Tensor, future: torch.Tensor) -> int:
    sequence = torch.cat((history[:, -1:], future), 1)
    motion = sequence.diff(dim=1).abs().mean(1)
    return int((motion[:, 7:].mean(1) > motion[:, :7].mean(1)).item())


@torch.no_grad()
def parent_trajectory_fp32(model, context, history, future, memory):
    """Match the established v19 full-dev evaluator exactly (no autocast)."""
    bases, warps = [], []
    for action in future.unbind(1):
        _, detail = model(context, torch.cat((history, action[:, None]), 1), memory, True)
        base = detail["base"].clamp(0, 1)
        bases.append(base)
        warps.append(detail["warped"].clamp(0, 1))
        context = torch.cat((context[:, 1:], base[:, None]), 1)
        history = torch.cat((history[:, 1:], action[:, None]), 1)
    return torch.stack(bases, 1), torch.stack(warps, 1)


def blank_accumulator():
    return {
        "rgb": torch.zeros(8), "texture": torch.zeros(8), "mask": torch.zeros(8),
        "arms": {0: [0., 0., 0., 0., 0.], 1: [0., 0., 0., 0., 0.]},
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--memory-checkpoint", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")

    split = json.loads(Path(args.split_manifest).read_text())
    data = WindowDataset(Path(args.windows), split["validation_episodes"])
    loader = DataLoader(data, batch_size=1, num_workers=2, pin_memory=True)
    parent, mean, std = load_parent(Path(args.init_checkpoint), Path(args.memory_checkpoint), device)
    selector = TrajectoryPatchSelector().to(device)
    selector_state = torch.load(Path(args.selector_checkpoint) / "selector.pt", map_location="cpu", weights_only=True)
    selector.load_state_dict(selector_state["state_dict"], strict=True)
    selector.eval().requires_grad_(False)

    # First three frames are exact parent fallback.  Later frames sweep a small
    # high-frequency-only correction and patch-confidence threshold.
    configs = [(confidence, alpha) for confidence in (0., .40, .55, .70, .80, .90)
               for alpha in (.01, .025, .05, .10, .20, .30)]
    names = [f"c{confidence:.2f}_a{alpha:.3f}" for confidence, alpha in configs]
    values = {name: blank_accumulator() for name in names}
    parent_rgb = torch.zeros(8)
    parent_texture = torch.zeros(8)
    selected_rgb = torch.zeros(8)
    selector_accuracy = 0.
    selector_switch = 0.
    count = 0

    for raw_context, raw_history, raw_future, raw_target in loader:
        active_arm = arm_index(raw_history, raw_future)
        context = frames_for_model(raw_context).to(device)
        target = frames_for_model(raw_target).to(device)
        history = ((raw_history.to(device) - mean) / std).float()
        future = ((raw_future.to(device) - mean) / std).float()
        base, warped = parent_trajectory_fp32(parent, context, history, future, context.clone())
        logits = selector(base.float(), warped.float(), future.float())
        probabilities = logits.softmax(2)
        source = probabilities.argmax(2)
        confidence = probabilities.max(2).values
        selector_switch += float((source[:, 1:] != source[:, :-1]).float().mean())

        weights_patch = F.one_hot(source, 5).permute(0, 1, 4, 2, 3).float()
        weights = F.interpolate(weights_patch.flatten(0, 1), size=base.shape[-2:], mode="nearest")
        weights = weights.reshape(1, 8, 5, *base.shape[-2:])
        selected = (warped * weights[:, :, :, None]).sum(2)
        selected_rgb += 255 * (selected - target).abs().mean((0, 2, 3, 4)).cpu()

        # Diagnostic accuracy uses the true lowest combined RGB/high-frequency
        # source per patch, but target information never enters rendering.
        target_hp = highpass(target.flatten(0, 1)).reshape_as(target)
        warped_hp = highpass(warped.flatten(0, 2)).reshape_as(warped)
        oracle_cost = (warped - target[:, :, None]).abs().mean(3)
        oracle_cost += 2 * (warped_hp - target_hp[:, :, None]).abs().mean(3)
        oracle_patch = F.avg_pool2d(oracle_cost.flatten(0, 2), 8, 8).reshape(1, 8, 5, 32, 32).argmin(2)
        selector_accuracy += float((source == oracle_patch).float().mean())

        parent_error = (base - target).abs().mean((2, 3, 4)).cpu()
        parent_hp_error = (highpass(base.flatten(0, 1)) - target_hp.flatten(0, 1)).abs().mean((1, 2, 3)).reshape(1, 8).cpu()
        parent_rgb += 255 * parent_error[0]
        parent_texture += 255 * parent_hp_error[0]
        high_delta = highpass(selected.flatten(0, 1)).reshape_as(selected) - highpass(base.flatten(0, 1)).reshape_as(base)

        for name, (threshold, alpha) in zip(names, configs):
            patch_mask = (confidence >= threshold).float()
            patch_mask[:, :3] = 0
            mask = F.interpolate(patch_mask.flatten(0, 1)[:, None], size=base.shape[-2:], mode="nearest")
            mask = mask.reshape(1, 8, 1, *base.shape[-2:])
            candidate = (base + alpha * mask * high_delta).clamp(0, 1)
            rgb = (candidate - target).abs().mean((2, 3, 4)).cpu()
            candidate_hp = highpass(candidate.flatten(0, 1)).reshape_as(candidate)
            texture = (candidate_hp - target_hp).abs().mean((2, 3, 4)).cpu()
            value = values[name]
            value["rgb"] += 255 * rgb[0]
            value["texture"] += 255 * texture[0]
            value["mask"] += mask.mean((0, 2, 3, 4)).cpu()
            record = value["arms"][active_arm]
            record[0] += float(255 * parent_error.mean())
            record[1] += float(255 * rgb.mean())
            record[2] += float(255 * parent_hp_error.mean())
            record[3] += float(255 * texture.mean())
            record[4] += 1
        count += 1
        if count % 50 == 0:
            print(json.dumps({"processed": count, "total": len(data)}), flush=True)

    parent_rgb /= count
    parent_texture /= count
    selected_rgb /= count
    results = []
    for name, (threshold, alpha) in zip(names, configs):
        value = values[name]
        rgb = value["rgb"] / count
        texture = value["texture"] / count
        frame_improvement = 100 * (parent_rgb - rgb) / parent_rgb
        item = {
            "name": name, "confidence": threshold, "alpha": alpha,
            "mask_rate": float(value["mask"].mean() / count),
            "rgb_mae": float(rgb.mean()), "texture_mae": float(texture.mean()),
            "rgb_improvement_percent": float(100 * (parent_rgb.mean() - rgb.mean()) / parent_rgb.mean()),
            "texture_improvement_percent": float(100 * (parent_texture.mean() - texture.mean()) / parent_texture.mean()),
            "frame_rgb_mae": rgb.tolist(), "frame_rgb_improvement_percent": frame_improvement.tolist(),
            "arms": {},
        }
        scores = [item["rgb_improvement_percent"], item["texture_improvement_percent"], float(frame_improvement.min())]
        for arm, (pr, vr, pt, vt, number) in value["arms"].items():
            arm_result = {
                "windows": int(number),
                "parent_rgb_mae": pr / number, "rgb_mae": vr / number,
                "parent_texture_mae": pt / number, "texture_mae": vt / number,
                "rgb_improvement_percent": 100 * (pr - vr) / pr,
                "texture_improvement_percent": 100 * (pt - vt) / pt,
            }
            item["arms"][f"arm{arm}"] = arm_result
            scores.extend((arm_result["rgb_improvement_percent"], arm_result["texture_improvement_percent"]))
        item["gate_score"] = min(scores)
        results.append(item)
    results.sort(key=lambda candidate: (candidate["gate_score"], candidate["texture_improvement_percent"]), reverse=True)
    report = {
        "format": "v21.1-trajectory-selector-high-frequency-renderer-calibration",
        "data_boundary": "supplied_50_episodes_only",
        "episodes": split["validation_episodes"], "windows": len(data),
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_unsmoothed_oracle_accuracy": selector_accuracy / count,
        "selector_switch_rate": selector_switch / count,
        "parent_rgb_mae": float(parent_rgb.mean()),
        "parent_texture_mae": float(parent_texture.mean()),
        "selected_source_rgb_mae": float(selected_rgb.mean()),
        "parent_frame_rgb_mae": parent_rgb.tolist(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selector": {"accuracy": selector_accuracy / count, "switch_rate": selector_switch / count,
                                            "selected_rgb_mae": float(selected_rgb.mean())},
                      "top": results[:8]}), flush=True)


if __name__ == "__main__":
    main()
