#!/usr/bin/env python3
"""Evaluate the deployable-flow candidate ceiling when anchored by a parent.

Unlike the RAFT transport oracle, all flows here are predicted from observations
and actions.  Ground truth is used only for the diagnostic block router.  This
separates the flow-prediction bottleneck from the learnable-routing bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet


def choose_blocks(candidates: torch.Tensor, target: torch.Tensor, block: int) -> torch.Tensor:
    # candidates [T,K,3,H,W], target [T,3,H,W]
    steps, count, _, height, width = candidates.shape
    error = (candidates - target[:, None]).abs().mean(dim=2)
    pooled = error.reshape(steps, count, height // block, block, width // block, block).mean(dim=(3, 5))
    choice = pooled.argmin(dim=1).repeat_interleave(block, 1).repeat_interleave(block, 2)
    return candidates.gather(1, choice[:, None, None].expand(-1, 1, 3, -1, -1)).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--include-window", action="append", default=[])
    parser.add_argument("--blocks", default="1,4,8,16,32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]
        names = cache["windows"].astype(str).tolist()
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = list(np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=np.int64))
    for name in args.include_window:
        if name not in name_to_index:
            raise ValueError(f"missing included window: {name}")
        indices.append(name_to_index[name])
    indices = list(dict.fromkeys(indices))
    blocks = [int(value) for value in args.blocks.split(",")]
    if any(value < 1 or 256 % value for value in blocks):
        raise ValueError("blocks must be positive divisors of 256")

    checkpoint = Path(args.flow_checkpoint)
    with np.load(checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        base_channels = int(config["base_channels"])
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model = MultiSourceActionFlowUNet(base_channels).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)

    methods = ["v8_parent", "old_multisource_fusion"] + [f"hybrid_block{block}_oracle" for block in blocks]
    sums = {name: 0.0 for name in methods}
    moving_sums = {name: 0.0 for name in methods}
    high_sums = {name: 0.0 for name in methods}
    total_count = moving_count = high_count = 0
    records = []
    for completed, index in enumerate(indices, start=1):
        name = names[index]
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            context_np = window["context_frames"]
            target_np = window["target_frames"]
            actions_np = np.concatenate((window["history_actions"], window["future_actions"]), axis=0)
        context = torch.from_numpy(context_np.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        target = torch.from_numpy(target_np.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        actions = torch.from_numpy(actions_np.copy()).to(device).float()
        normalized = (actions - action_mean) / action_std
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fused, flow, _ = model(context[None], normalized[None], return_flow=True)
            warped = model._warp(context[None], flow).squeeze(0)
        fused = fused.squeeze(0).float().clamp(0, 1)
        warped = warped.float().clamp(0, 1)
        baseline = torch.from_numpy(parent[index].copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        candidates = torch.cat((baseline[:, None], warped, fused[:, None]), dim=1)
        predictions = {"v8_parent": baseline, "old_multisource_fusion": fused}
        predictions.update({f"hybrid_block{block}_oracle": choose_blocks(candidates, target, block) for block in blocks})
        previous = torch.cat((context[-1:], target[:-1]), dim=0)
        moving = (target - previous).abs().mean(dim=1) >= 0.03
        motion_score = float((target - previous).abs().mean())
        high_motion = motion_score >= 0.04
        for key, value in predictions.items():
            error = (value - target).abs()
            sums[key] += float(error.sum())
            moving_sums[key] += float((error * moving[:, None]).sum())
            if high_motion:
                high_sums[key] += float(error.sum())
        total_count += target.numel()
        moving_count += int(moving.sum()) * 3
        if high_motion:
            high_count += target.numel()
        records.append({"window": name, "motion_score": motion_score})
        print(json.dumps({"completed": completed, "total": len(indices), "window": name}), flush=True)

    metrics = {
        key: {
            "rgb_mae_0_255": 255.0 * sums[key] / total_count,
            "moving_rgb_mae_0_255": 255.0 * moving_sums[key] / moving_count,
            "high_motion_rgb_mae_0_255": 255.0 * high_sums[key] / high_count if high_count else None,
        }
        for key in methods
    }
    baseline_mae = metrics["v8_parent"]["rgb_mae_0_255"]
    for value in metrics.values():
        value["relative_rgb_improvement_over_v8_percent"] = 100.0 * (baseline_mae - value["rgb_mae_0_255"]) / baseline_mae
    result = {
        "format": "track2-predicted-multisource-hybrid-oracle-v1",
        "warning": "Flow is deployable, but block routing uses the future target and is an oracle.",
        "sample_count": len(indices),
        "high_motion_sample_count": sum(record["motion_score"] >= 0.04 for record in records),
        "candidate_count": 7,
        "flow_checkpoint": str(checkpoint.resolve()),
        "metrics": metrics,
        "samples": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
