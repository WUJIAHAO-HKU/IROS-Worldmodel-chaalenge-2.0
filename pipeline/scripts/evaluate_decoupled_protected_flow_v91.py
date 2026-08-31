#!/usr/bin/env python3
"""Combine a flow-specialist v9.1 head with a confidence-specialist head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet
from wam_pipeline.protected_layered_flow_v91 import ProtectedLayeredFlowV91


def build_model(base_checkpoint: Path, head_checkpoint: Path, device: torch.device):
    with np.load(base_checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        base = MultiSourceActionFlowUNet(int(config["base_channels"]))
    base_state = torch.load(base_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    base.load_state_dict(base_state["state_dict"], strict=True)
    head = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    model = ProtectedLayeredFlowV91(base, int(head["base_channels"]), float(head["max_residual_flow"]))
    incompatible = model.load_state_dict(head["state_dict"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("base_flow_model.")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"bad head {head_checkpoint}: {missing} {incompatible.unexpected_keys}")
    return model.to(device).eval(), head["active_mean"].to(device), head["active_std"].to(device), int(head["step"])


def frames(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--confidence-head", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--selection", choices=("top-motion", "uniform"), default="top-motion")
    parser.add_argument("--fixed-margin", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    base_checkpoint = Path(args.base_checkpoint)
    flow_model, flow_active_mean, flow_active_std, flow_step = build_model(base_checkpoint, Path(args.flow_head), device)
    confidence_model, confidence_active_mean, confidence_active_std, confidence_step = build_model(base_checkpoint, Path(args.confidence_head), device)
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
        score_by_index = {index: score for score, index in motion}
        selected = [(score_by_index[int(index)], int(index)) for index in indices]
    margins = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
    route_sum = {margin: 0.0 for margin in margins}
    route_moving = {margin: 0.0 for margin in margins}
    selection = {margin: 0 for margin in margins}
    baseline_sum = oracle_sum = baseline_moving = oracle_moving = 0.0
    total = moving_count = block_count = 0
    records = []
    with torch.inference_mode():
        for score, index in selected:
            name = names[index]
            with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                context = frames(window["context_frames"], device)[None]
                target = frames(window["target_frames"], device)[None]
                actions_np = np.concatenate((window["history_actions"], window["future_actions"]), axis=0)
            parent = frames(parent_cache[index], device)[None]
            actions = torch.from_numpy(actions_np.copy()).to(device).float()[None]
            active, arm = flow_model.active_arm_actions(actions)
            flow_result = flow_model(context, (actions - action_mean) / action_std,
                                     (active - flow_active_mean) / flow_active_std, arm, parent)
            confidence_result = confidence_model(context, (actions - action_mean) / action_std,
                                                 (active - confidence_active_mean) / confidence_active_std, arm, parent)
            candidates = flow_result["candidates"]
            logits = confidence_result["route_logits"]
            error = (candidates - target[:, :, None]).abs().mean(dim=3)
            pooled = torch.nn.functional.avg_pool2d(error.flatten(0, 2)[:, None], 4, stride=4).squeeze(1)
            pooled = pooled.unflatten(0, error.shape[:3])
            oracle_choice = pooled.argmin(dim=2)
            full_oracle = oracle_choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            oracle = candidates.gather(2, full_oracle[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
            moving = (target - previous).abs().mean(dim=2) >= 0.03
            base_error = (parent - target).abs(); oracle_error = (oracle - target).abs()
            baseline_sum += float(base_error.sum()); oracle_sum += float(oracle_error.sum())
            baseline_moving += float((base_error * moving[:, :, None]).sum())
            oracle_moving += float((oracle_error * moving[:, :, None]).sum())
            other_score, other_choice = logits[:, :, 1:].max(dim=2)
            window_routes = {}
            for margin in margins:
                choice = torch.where(other_score - logits[:, :, 0] > margin, other_choice + 1, 0)
                full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
                routed = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
                routed_error = (routed - target).abs()
                route_sum[margin] += float(routed_error.sum())
                route_moving[margin] += float((routed_error * moving[:, :, None]).sum())
                selection[margin] += int((choice != 0).sum())
                window_routes[str(margin)] = float(routed_error.mean() * 255)
            total += target.numel(); moving_count += int(moving.sum()) * 3
            block_count += logits.shape[0] * logits.shape[1] * logits.shape[3] * logits.shape[4]
            records.append({"window": name, "motion_score": score, "baseline_rgb_mae": float(base_error.mean() * 255),
                            "oracle_rgb_mae": float(oracle_error.mean() * 255), "route_rgb_mae": window_routes})
    if args.fixed_margin is not None:
        if args.fixed_margin not in margins:
            raise ValueError(f"fixed margin must be one of {margins}")
        best_margin = args.fixed_margin
    else:
        best_margin = min(margins, key=lambda margin: route_sum[margin])
    baseline = 255 * baseline_sum / total
    router = 255 * route_sum[best_margin] / total
    oracle = 255 * oracle_sum / total
    result = {
        "format": "track2-decoupled-protected-layered-flow-v9.1-eval",
        "sample_count": len(selected), "selection": args.selection, "fixed_margin": args.fixed_margin,
        "flow_head": str(Path(args.flow_head).resolve()), "flow_head_step": flow_step,
        "confidence_head": str(Path(args.confidence_head).resolve()), "confidence_head_step": confidence_step,
        "baseline_rgb_mae": baseline, "block4_oracle_rgb_mae": oracle,
        "block4_oracle_improvement_percent": 100 * (baseline - oracle) / baseline,
        "router_rgb_mae": router, "router_improvement_percent": 100 * (baseline - router) / baseline,
        "best_margin": best_margin, "margin_rgb_mae": {str(margin): 255 * route_sum[margin] / total for margin in margins},
        "moving_baseline_rgb_mae": 255 * baseline_moving / moving_count,
        "moving_oracle_rgb_mae": 255 * oracle_moving / moving_count,
        "moving_router_rgb_mae": 255 * route_moving[best_margin] / moving_count,
        "selected_transport_block_fraction": selection[best_margin] / block_count,
        "hard_gate": {"candidate_oracle_pass": 100 * (baseline - oracle) / baseline >= 35,
                      "router_pass": 100 * (baseline - router) / baseline >= 5},
        "windows": records,
    }
    result["hard_gate"]["passed"] = result["hard_gate"]["candidate_oracle_pass"] and result["hard_gate"]["router_pass"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
