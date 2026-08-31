#!/usr/bin/env python3
"""Evaluate a dev-calibrated v10 candidate risk router on a separate cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.candidate_risk_router_v10 import CandidateRiskRouterV10
from wam_pipeline.candidate_risk_router_v101 import CandidateRiskRouterV101
from train_candidate_risk_router_v10 import RiskDataset, _auc, frames, load_flow, pool_error


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    smooth = F.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
    return (flat - smooth).unflatten(0, value.shape[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--selection", choices=("uniform", "top-motion"), default="uniform")
    parser.add_argument("--risk-weight", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--highpass-protection", type=float, default=0.0)
    parser.add_argument("--checkpoint-dev-episodes", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-output", help="Optional NPZ containing routed predictions and targets.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 <= args.highpass_protection <= 1.0:
        raise ValueError("highpass-protection must be in [0,1]")
    device = torch.device(args.device)
    # This checkpoint is produced locally by the paired trainer and includes
    # NumPy scalar metrics in addition to tensor weights.
    checkpoint = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=False)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    motion = []
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            previous = np.concatenate((window["context_frames"][-1:].astype(np.float32), target[:-1]), axis=0)
        motion.append((float(np.abs(target - previous).mean() / 255.0), index))
    if args.checkpoint_dev_episodes:
        dev_episodes = {str(value) for value in checkpoint["dev_episodes"]}
        indices = [index for index, name in enumerate(names) if name.split("_")[0] in dev_episodes]
        if len(indices) > args.samples:
            positions = np.linspace(0, len(indices) - 1, args.samples, dtype=np.int64)
            indices = [indices[int(position)] for position in positions]
    elif args.selection == "top-motion":
        indices = [index for _, index in sorted(motion, reverse=True)[: args.samples]]
    else:
        indices = np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=np.int64).tolist()
    dataset = RiskDataset(Path(args.windows), parent, names, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    base_checkpoint = Path(args.base_checkpoint)
    flow_model, active_mean, active_std = load_flow(base_checkpoint, Path(args.flow_head), device)
    with np.load(base_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)
    router_version = str(checkpoint.get("router_version", "v10"))
    router_class = CandidateRiskRouterV101 if router_version == "v10.1" else CandidateRiskRouterV10
    router = router_class(int(checkpoint["base_channels"]))
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    router = router.to(device).eval()
    risk_weight = checkpoint["metrics"]["best_risk_weight"] if args.risk_weight is None else args.risk_weight
    margin = checkpoint["metrics"]["best_margin"] if args.margin is None else args.margin

    baseline_sum = oracle_sum = routed_sum = 0.0
    moving_baseline = moving_oracle = moving_routed = 0.0
    highpass_baseline = highpass_routed = dark_baseline = dark_routed = 0.0
    edge_baseline = edge_routed = edge_count = dark_count = 0.0
    total = moving_count = block_count = selected_count = true_positive = beneficial_count = 0
    auc_labels, auc_scores, records = [], [], []
    saved_predictions, saved_targets, saved_contexts, saved_choices, saved_names = [], [], [], [], []
    with torch.inference_mode():
        for parent_batch, context, history, future, target, batch_names in loader:
            parent_batch, context, target = (frames(value, device) for value in (parent_batch, context, target))
            actions = torch.cat((history, future), dim=1).to(device).float()
            active, arm = flow_model.active_arm_actions(actions)
            flow = flow_model(context, (actions - action_mean) / action_std,
                              (active - active_mean) / active_std, arm, parent_batch)
            candidates = flow["candidates"]
            risk = router(context, parent_batch, candidates[:, :, 1:], flow["refined_flow"],
                          flow["visibility_logits"], active, arm)
            error = pool_error(candidates, target)
            advantage = error[:, :, :1] - error[:, :, 1:]
            choice, _ = router.safe_choice(
                risk["advantage_mean"], risk["advantage_uncertainty"], risk_weight, margin
            )
            full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            routed = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            if args.highpass_protection:
                routed = (routed + args.highpass_protection
                          * (highpass(parent_batch) - highpass(routed))).clamp(0, 1)
            oracle_choice = error.argmin(dim=2)
            full_oracle = oracle_choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            oracle = candidates.gather(
                2, full_oracle[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)
            ).squeeze(2)
            previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
            moving = (target - previous).abs().mean(dim=2) >= 0.03
            base_error = (parent_batch - target).abs()
            oracle_error = (oracle - target).abs(); route_error = (routed - target).abs()
            baseline_sum += float(base_error.sum()); oracle_sum += float(oracle_error.sum())
            routed_sum += float(route_error.sum())
            moving_baseline += float((base_error * moving[:, :, None]).sum())
            moving_oracle += float((oracle_error * moving[:, :, None]).sum())
            moving_routed += float((route_error * moving[:, :, None]).sum())
            base_highpass_error = (highpass(parent_batch) - highpass(target)).abs()
            route_highpass_error = (highpass(routed) - highpass(target)).abs()
            highpass_baseline += float(base_highpass_error.sum())
            highpass_routed += float(route_highpass_error.sum())
            dark = target.mean(dim=2) < 0.30
            dark_baseline += float((base_error * dark[:, :, None]).sum())
            dark_routed += float((route_error * dark[:, :, None]).sum())
            dark_count += int(dark.sum()) * 3
            for dimension in (-1, -2):
                base_gradient = torch.diff(parent_batch, dim=dimension)
                route_gradient = torch.diff(routed, dim=dimension)
                target_gradient = torch.diff(target, dim=dimension)
                edge_baseline += float((base_gradient - target_gradient).abs().sum())
                edge_routed += float((route_gradient - target_gradient).abs().sum())
                edge_count += target_gradient.numel()
            selected = choice != 0
            selected_source = (choice - 1).clamp_min(0)
            realized = advantage.gather(2, selected_source[:, :, None]).squeeze(2)
            beneficial = advantage.max(dim=2).values > 0.5
            selected_count += int(selected.sum()); true_positive += int((selected & (realized > 0.5)).sum())
            beneficial_count += int(beneficial.sum())
            auc_labels.append(beneficial.flatten()[::32].cpu().numpy())
            auc_scores.append(risk["advantage_mean"].max(dim=2).values.flatten()[::32].cpu().numpy())
            total += target.numel(); moving_count += int(moving.sum()) * 3
            block_count += error.shape[0] * error.shape[1] * 64 * 64
            records.append({
                "window": batch_names[0], "baseline_rgb_mae": float(base_error.mean() * 255),
                "oracle_rgb_mae": float(oracle_error.mean() * 255),
                "router_rgb_mae": float(route_error.mean() * 255),
                "selected_block_fraction": float(selected.float().mean()),
            })
            if args.prediction_output:
                saved_predictions.append(
                    routed.mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()[0]
                )
                saved_targets.append(
                    target.mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()[0]
                )
                saved_contexts.append(
                    context.mul(255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()[0]
                )
                saved_choices.append(choice.byte().cpu().numpy()[0])
                saved_names.append(batch_names[0])
    baseline = 255 * baseline_sum / total
    routed_mae = 255 * routed_sum / total
    result = {
        "format": "track2-candidate-risk-router-v10-eval", "sample_count": len(dataset),
        "selection": args.selection, "risk_checkpoint": str(Path(args.risk_checkpoint).resolve()),
        "risk_step": int(checkpoint["step"]), "router_version": router_version,
        "risk_weight": float(risk_weight), "margin": float(margin),
        "highpass_protection": float(args.highpass_protection),
        "checkpoint_dev_episodes": bool(args.checkpoint_dev_episodes),
        "baseline_rgb_mae": baseline, "block4_oracle_rgb_mae": 255 * oracle_sum / total,
        "block4_oracle_improvement_percent": 100 * (baseline_sum - oracle_sum) / baseline_sum,
        "router_rgb_mae": routed_mae, "router_improvement_percent": 100 * (baseline - routed_mae) / baseline,
        "moving_baseline_rgb_mae": 255 * moving_baseline / moving_count,
        "moving_oracle_rgb_mae": 255 * moving_oracle / moving_count,
        "moving_router_rgb_mae": 255 * moving_routed / moving_count,
        "highpass_baseline_mae": 255 * highpass_baseline / total,
        "highpass_router_mae": 255 * highpass_routed / total,
        "edge_baseline_mae": 255 * edge_baseline / edge_count,
        "edge_router_mae": 255 * edge_routed / edge_count,
        "dark_baseline_rgb_mae": 255 * dark_baseline / dark_count,
        "dark_router_rgb_mae": 255 * dark_routed / dark_count,
        "selected_transport_block_fraction": selected_count / block_count,
        "selected_precision": true_positive / max(selected_count, 1),
        "beneficial_recall": true_positive / max(beneficial_count, 1),
        "beneficial_block_auroc": _auc(auc_labels, auc_scores), "windows": records,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.prediction_output:
        prediction_output = Path(args.prediction_output)
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_output,
            prediction=np.stack(saved_predictions),
            target=np.stack(saved_targets),
            context=np.stack(saved_contexts),
            choice=np.stack(saved_choices),
            windows=np.asarray(saved_names),
        )
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2))


if __name__ == "__main__":
    main()
