#!/usr/bin/env python3
"""Evaluate a bounded post-V15 residual on an immutable frozen-V15 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_strict_track2_autoregressive_candidate import Accumulator, gains, metric_rows
from train_strict_track2_post_v15_residual import CachedSequences, active_actions, frames, rollout
from wam_pipeline.post_v15_residual import PostV15ResidualUNet
from wam_pipeline.visual_source_gate_runtime import Track2VisualSourceGate


CONTACT_REGION = (72, 232, 30, 210)


def spatial_mask(policy: str, baseline: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    if policy == "none":
        return torch.ones_like(baseline[:, :, :1])
    batch, time, _, height, width = baseline.shape
    y0, y1, x0, x1 = CONTACT_REGION
    y = torch.arange(y1 - y0, device=baseline.device)
    x = torch.arange(x1 - x0, device=baseline.device)
    distance_y = torch.minimum(y + 1, y1 - y0 - y)
    distance_x = torch.minimum(x + 1, x1 - x0 - x)
    feather = torch.minimum(distance_y[:, None], distance_x[None]).to(baseline.dtype).div(16).clamp(0, 1)
    contact = torch.zeros((1, 1, 1, height, width), device=baseline.device, dtype=baseline.dtype)
    contact[:, :, :, y0:y1, x0:x1] = feather
    contact = contact.expand(batch, time, -1, -1, -1)
    if policy == "contact_feather16":
        return contact
    if policy in ("contact_motion_dilate9", "contact_hard_motion_dilate9"):
        motion = (baseline - context[:, None]).abs().mean(2, keepdim=True) > (2 / 255)
        motion = F.max_pool2d(
            motion.flatten(0, 1).float(), 9, 1, 4
        ).unflatten(0, (batch, time)).to(baseline.dtype)
        if policy == "contact_motion_dilate9":
            return contact * motion
        hard_contact = torch.zeros_like(contact)
        hard_contact[:, :, :, y0:y1, x0:x1] = 1
        return hard_contact * motion
    raise ValueError(f"unsupported mask policy: {policy}")


def load_checkpoint(path: Path, device: torch.device):
    state = torch.load(path / "post_v15_residual.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "strict-track2-post-v15-residual-v1":
        raise ValueError("unsupported post-V15 residual checkpoint")
    config = state["config"]
    model = PostV15ResidualUNet(
        int(config["base_channels"]), float(config["maximum_residual_255"])
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, state["active_action_mean"].to(device), state["active_action_std"].to(device), config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reward-cache-output")
    parser.add_argument(
        "--mask-policy",
        choices=("none", "contact_feather16", "contact_motion_dilate9", "contact_hard_motion_dilate9"),
        default="none",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--advantage-mask")
    parser.add_argument("--source-gate")
    parser.add_argument("--output-strength", type=float, default=1.0)
    parser.add_argument("--maximum-deployed-residual-255", type=float, default=8.0)
    args = parser.parse_args()
    if args.output_strength <= 0 or args.maximum_deployed_residual_255 <= 0:
        raise ValueError("strength and deployed residual bound must be positive")

    dataset = CachedSequences(Path(args.cache))
    device = torch.device(args.device)
    model, mean, std, config = load_checkpoint(Path(args.checkpoint), device)
    advantage_mask = None
    source_gate = None
    if args.advantage_mask:
        if not args.source_gate:
            raise ValueError("--source-gate is required with --advantage-mask")
        with np.load(args.advantage_mask, allow_pickle=False) as values:
            if str(values["format"]) != "strict-track2-residual-advantage-mask-v1":
                raise ValueError("unsupported residual advantage mask")
            advantage_mask = values["mask"].astype(np.float32)
        source_gate = Track2VisualSourceGate(args.source_gate, args.device)
    before, after = Accumulator(), Accumulator()
    records = {key: [] for key in (
        "context_last", "target", "baseline", "candidate", "arm_right",
        "capture_success", "path", "synthetic_seed", "start",
    )}
    max_observed_residual = 0.0
    gate_sum, gate_count = 0.0, 0
    source_routes = {"official": 0, "synthetic": 0}
    with torch.inference_mode():
        for index in range(len(dataset)):
            raw_context, raw_future, raw_baseline, raw_target, arm_right, _ = dataset[index]
            context = frames(raw_context[None, None], device)[:, 0]
            baseline = frames(raw_baseline[None], device)
            target = frames(raw_target[None], device)
            future = raw_future[None].to(device)
            arm_tensor = arm_right[None].to(device).bool()
            actions = active_actions(future, arm_tensor)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, detail = rollout(model, baseline, context, actions, arm_tensor, mean, std)
            prediction = prediction.float()
            correction = args.output_strength * (prediction - baseline)
            bound = args.maximum_deployed_residual_255 / 255
            prediction = baseline + correction.clamp(-bound, bound)
            mask = spatial_mask(args.mask_policy, baseline, context)
            prediction = baseline + mask * (prediction - baseline)
            if advantage_mask is not None:
                source_index, probability = source_gate.source_index(raw_context.numpy())
                arm_index = int(bool(arm_right))
                learned_mask = torch.from_numpy(advantage_mask[source_index, arm_index])
                learned_mask = learned_mask.to(device)[:, None][None]
                prediction = baseline + learned_mask * (prediction - baseline)
                source_routes["synthetic" if source_index else "official"] += 1
            candidate_uint8 = (
                prediction[0].mul(255).round().clamp(0, 255).byte()
                .permute(0, 2, 3, 1).cpu().numpy()
            )
            deployed_prediction = (
                torch.from_numpy(candidate_uint8).permute(0, 3, 1, 2)
                .to(device).float().div(255)[None]
            )
            baseline_rows = metric_rows(baseline, target, context)
            candidate_rows = metric_rows(deployed_prediction, target, context)
            record = dataset.records[index]
            right = torch.tensor([record["arm"] == "right"])
            synthetic = torch.tensor([record["source"] == "synthetic"])
            success = torch.tensor([bool(record["capture_success"])])
            masks = {
                "overall": torch.ones(1, dtype=torch.bool),
                "left": ~right,
                "right": right,
                "official": ~synthetic,
                "synthetic": synthetic,
                "capture_success": success,
                "capture_failure": ~success,
            }
            for group, mask in masks.items():
                before.add(group, baseline_rows, mask)
                after.add(group, candidate_rows, mask)
            residual = (deployed_prediction - baseline).abs()
            max_observed_residual = max(max_observed_residual, float(residual.max().cpu()) * 255)
            gate_sum += float(detail["gate"].sum().cpu())
            gate_count += detail["gate"].numel()
            with np.load(dataset.paths[index], allow_pickle=False) as cached:
                values = {
                    "context_last": cached["context_last"].copy(),
                    "target": cached["target"].copy(),
                    "baseline": cached["baseline"].copy(),
                    "candidate": candidate_uint8,
                    "arm_right": cached["arm_right"].copy(),
                    "capture_success": cached["capture_success"].copy(),
                    "path": str(record["path"]),
                    "synthetic_seed": cached["synthetic_seed"].copy(),
                    "start": cached["start"].copy(),
                }
            for key, value in values.items():
                records[key].append(value)
            if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(dataset):
                print(json.dumps({"evaluated": index + 1, "total": len(dataset)}), flush=True)

    baseline_result, candidate_result = before.result(), after.result()
    report = {
        "format": "strict-track2-post-v15-residual-evaluation-v1",
        "cache": str(Path(args.cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_config": config,
        "deployment_quantization": "round_clamp_uint8_before_all_visual_metrics",
        "mask_policy": args.mask_policy,
        "output_strength": args.output_strength,
        "maximum_deployed_residual_255": args.maximum_deployed_residual_255,
        "advantage_mask": str(Path(args.advantage_mask).resolve()) if args.advantage_mask else None,
        "source_gate": str(Path(args.source_gate).resolve()) if args.source_gate else None,
        "source_routes": source_routes,
        "selected_window_count": len(dataset),
        "maximum_observed_absolute_residual_255": max_observed_residual,
        "mean_gate": gate_sum / gate_count,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "improvement": gains(baseline_result, candidate_result),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.reward_cache_output:
        reward_output = Path(args.reward_cache_output)
        reward_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(reward_output, **{
            key: np.asarray(value) if key == "path" else np.stack(value)
            for key, value in records.items()
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
