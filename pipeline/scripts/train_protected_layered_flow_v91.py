#!/usr/bin/env python3
"""Train the v9.1 protected high-resolution flow/visibility model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet
from wam_pipeline.protected_layered_flow_v91 import ProtectedLayeredFlowV91


class PilotDataset(Dataset):
    def __init__(self, windows: Path, flow_targets: Path, parent: np.ndarray, names: list[str], indices: list[int]) -> None:
        self.windows, self.flow_targets, self.parent, self.names, self.indices = windows, flow_targets, parent, names, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            values = (
                self.parent[index], window["context_frames"].copy(), window["history_actions"].copy(),
                window["future_actions"].copy(), window["target_frames"].copy(),
            )
        teacher = np.load(self.flow_targets / f"{Path(name).stem}.npy", allow_pickle=False).astype(np.float32)
        if teacher.shape != (8, 5, 2, 128, 128):
            raise ValueError(f"invalid teacher flow for {name}: {teacher.shape}")
        return (*values, teacher, name)


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def active_statistics(dataset: PilotDataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for item in range(len(dataset)):
        _, _, history, future, _, _, _ = dataset[item]
        actions = torch.from_numpy(np.concatenate((history, future), axis=0))[None]
        active, _ = ProtectedLayeredFlowV91.active_arm_actions(actions)
        values.append(active.squeeze(0))
    value = torch.cat(values)
    return value.mean(0), value.std(0).clamp_min(1e-5)


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    return (flat - F.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)).unflatten(0, value.shape[:2])


def pool_candidate_error(candidates: torch.Tensor, target: torch.Tensor, block: int = 4) -> torch.Tensor:
    b, t, k = candidates.shape[:3]
    error = (candidates - target[:, :, None]).abs().mean(dim=3)
    pooled = F.avg_pool2d(error.flatten(0, 2)[:, None], block, stride=block).squeeze(1)
    return pooled.unflatten(0, (b, t, k))


def teacher_visibility(base_model, context, target, teacher_flow):
    b, t, sources = teacher_flow.shape[:3]
    context128 = F.interpolate(context.flatten(0, 1), (128, 128), mode="bilinear", align_corners=False).unflatten(0, (b, 5))
    target128 = F.interpolate(target.flatten(0, 1), (128, 128), mode="bilinear", align_corners=False).unflatten(0, (b, t))
    teacher_warp = base_model._warp(context128, teacher_flow)
    error = (teacher_warp - target128[:, :, None]).abs().mean(dim=3)
    y, x = torch.meshgrid(
        torch.arange(128, device=teacher_flow.device, dtype=teacher_flow.dtype),
        torch.arange(128, device=teacher_flow.device, dtype=teacher_flow.dtype), indexing="ij",
    )
    coordinates = torch.stack((x, y))[None, None, None]
    mapped = coordinates + teacher_flow
    inside = (mapped[:, :, :, 0] >= 0) & (mapped[:, :, :, 0] <= 127)
    inside &= (mapped[:, :, :, 1] >= 0) & (mapped[:, :, :, 1] <= 127)
    visible = inside & (error <= 10.0 / 255.0)
    return visible.to(target.dtype), error


def training_loss(result, context, target, teacher_flow, base_model, weights):
    prediction, candidates = result["prediction"], result["candidates"]
    b, t, k = candidates.shape[:3]
    previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
    motion = (target - previous).abs().mean(dim=2, keepdim=True)
    dark = target.mean(dim=2, keepdim=True) < 0.30
    image_weight = 1.0 + 3.0 * (motion >= 0.03) + 1.5 * dark
    pixel = (image_weight * (prediction - target).abs()).mean()
    texture = (image_weight * (highpass(prediction) - highpass(target)).abs()).mean()
    edge_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    edge = 0.5 * (edge_x.mean() + edge_y.mean())

    refined = result["refined_flow"]
    refined128 = F.interpolate(refined.flatten(0, 2), (128, 128), mode="bilinear", align_corners=True)
    refined128 = refined128.unflatten(0, refined.shape[:3]) * 0.5
    flow_weight = 1.0 + (teacher_flow.square().sum(dim=3, keepdim=True).sqrt() / 8.0).clamp(0, 4)
    flow = (flow_weight * F.smooth_l1_loss(refined128 / 16.0, teacher_flow / 16.0, reduction="none")).mean()
    base128 = F.interpolate(result["base_flow"].flatten(0, 2), (128, 128), mode="bilinear", align_corners=True)
    base128 = base128.unflatten(0, result["base_flow"].shape[:3]) * 0.5
    base_flow_error = F.l1_loss(base128, teacher_flow).detach()
    refined_flow_error = F.l1_loss(refined128, teacher_flow).detach()

    visible, teacher_error = teacher_visibility(base_model, context, target, teacher_flow)
    visibility_logits = F.interpolate(
        result["visibility_logits"].flatten(0, 2)[:, None], (128, 128), mode="bilinear", align_corners=False
    ).squeeze(1).unflatten(0, (b, t, 5))
    positive = visible.mean().clamp_min(1e-4)
    positive_weight = ((1 - positive) / positive).clamp(1, 8)
    visibility = F.binary_cross_entropy_with_logits(visibility_logits, visible, pos_weight=positive_weight)

    block_error = pool_candidate_error(candidates, target)
    parent_error = block_error[:, :, :1]
    best_other_error, best_other = block_error[:, :, 1:].min(dim=2)
    target_choice = torch.where(best_other_error + 0.5 / 255.0 < parent_error.squeeze(2), best_other + 1, 0)
    route_logits = result["route_logits"]
    classification = F.cross_entropy(route_logits.flatten(0, 1), target_choice.flatten(0, 1), reduction="none")
    moving64 = F.avg_pool2d(motion.flatten(0, 1), 4, stride=4).unflatten(0, (b, t)).squeeze(2)
    classification = (classification.unflatten(0, (b, t)) * (1 + 2 * (moving64 >= 0.03))).mean()
    target_advantage = ((parent_error - block_error) * 25.5).clamp(-5, 5)
    predicted_advantage = route_logits - route_logits[:, :, :1]
    advantage_error = F.smooth_l1_loss(predicted_advantage[:, :, 1:], target_advantage[:, :, 1:], beta=0.25, reduction="none")
    advantage_weight = 1 + 3 * (target_advantage[:, :, 1:] > 0).to(advantage_error.dtype)
    advantage_weight = advantage_weight * (1 + moving64[:, :, None])
    advantage = (advantage_error * advantage_weight).sum() / advantage_weight.sum().clamp_min(1)
    expected = (result["route_weights"] * block_error).sum(dim=2).mean()
    # Optimize the best currently available transported source independently
    # of confidence routing, using detached soft assignments for stability.
    transport_error = block_error[:, :, 1:]
    transport_assignment = torch.softmax(-transport_error / 0.01, dim=2).detach()
    candidate_softmin = (transport_assignment * transport_error).sum(dim=2).mean()

    residual = result["residual_flow"]
    smooth_x = (residual[..., 1:] - residual[..., :-1]).abs().mean()
    smooth_y = (residual[..., 1:, :] - residual[..., :-1, :]).abs().mean()
    residual_regularizer = residual.abs().mean() / 6.0 + 0.25 * (smooth_x + smooth_y) / 6.0
    total = (
        pixel + weights["texture"] * texture + weights["edge"] * edge + weights["flow"] * flow
        + weights["visibility"] * visibility + weights["classification"] * classification
        + weights["advantage"] * advantage + weights["expected"] * expected
        + weights["candidate_softmin"] * candidate_softmin
        + weights["residual"] * residual_regularizer
    )
    parts = {
        "pixel": pixel, "texture": texture, "edge": edge, "flow": flow, "visibility": visibility,
        "classification": classification, "advantage": advantage, "expected": expected,
        "candidate_softmin": candidate_softmin,
        "residual_regularizer": residual_regularizer, "base_flow_epe": base_flow_error,
        "refined_flow_epe": refined_flow_error, "teacher_visible_fraction": visible.mean(),
        "teacher_warp_error": teacher_error.mean(),
    }
    return total, parts


@torch.inference_mode()
def evaluate(loader, model, device, flow_mean, flow_std, active_mean, active_std):
    margins = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    sums = {margin: 0.0 for margin in margins}
    baseline_sum = oracle_sum = moving_baseline = moving_oracle = 0.0
    moving_sums = {margin: 0.0 for margin in margins}
    selection = {margin: 0 for margin in margins}
    total = moving_count = block_count = 0
    flow_base = flow_refined = flow_count = 0.0
    model.eval()
    for parent, context, history, future, target, teacher, _ in loader:
        parent, context, target = (frames(value, device) for value in (parent, context, target))
        actions = torch.cat((history, future), dim=1).to(device).float()
        teacher = teacher.to(device).float()
        active, arm = model.active_arm_actions(actions)
        result = model(context, (actions - flow_mean) / flow_std, (active - active_mean) / active_std, arm, parent)
        candidates, logits = result["candidates"], result["route_logits"]
        error = pool_candidate_error(candidates, target)
        oracle_choice = error.argmin(dim=2)
        full_oracle = oracle_choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
        oracle = candidates.gather(2, full_oracle[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
        moving = (target - previous).abs().mean(dim=2) >= 0.03
        baseline_error = (parent - target).abs(); oracle_error = (oracle - target).abs()
        baseline_sum += float(baseline_error.sum()); oracle_sum += float(oracle_error.sum())
        moving_baseline += float((baseline_error * moving[:, :, None]).sum())
        moving_oracle += float((oracle_error * moving[:, :, None]).sum())
        other_score, other_choice = logits[:, :, 1:].max(dim=2)
        for margin in margins:
            choice = torch.where(other_score - logits[:, :, 0] > margin, other_choice + 1, 0)
            full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            routed = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            routed_error = (routed - target).abs()
            sums[margin] += float(routed_error.sum())
            moving_sums[margin] += float((routed_error * moving[:, :, None]).sum())
            selection[margin] += int((choice != 0).sum())
        refined128 = F.interpolate(result["refined_flow"].flatten(0, 2), (128, 128), mode="bilinear", align_corners=True)
        refined128 = refined128.unflatten(0, result["refined_flow"].shape[:3]) * 0.5
        base128 = F.interpolate(result["base_flow"].flatten(0, 2), (128, 128), mode="bilinear", align_corners=True)
        base128 = base128.unflatten(0, result["base_flow"].shape[:3]) * 0.5
        flow_base += float((base128 - teacher).abs().sum()); flow_refined += float((refined128 - teacher).abs().sum())
        flow_count += teacher.numel(); total += target.numel(); moving_count += int(moving.sum()) * 3
        block_count += logits.shape[0] * logits.shape[1] * logits.shape[3] * logits.shape[4]
    baseline = 255 * baseline_sum / total
    margin_metrics = {str(margin): 255 * sums[margin] / total for margin in margins}
    best_margin = min(margins, key=lambda margin: sums[margin])
    return {
        "baseline_rgb_mae": baseline, "block4_oracle_rgb_mae": 255 * oracle_sum / total,
        "block4_oracle_improvement_percent": 100 * (baseline_sum - oracle_sum) / baseline_sum,
        "router_rgb_mae": margin_metrics[str(best_margin)], "best_margin": best_margin,
        "router_improvement_percent": 100 * (baseline_sum - sums[best_margin]) / baseline_sum,
        "margin_rgb_mae": margin_metrics,
        "moving_baseline_rgb_mae": 255 * moving_baseline / moving_count,
        "moving_oracle_rgb_mae": 255 * moving_oracle / moving_count,
        "moving_router_rgb_mae": 255 * moving_sums[best_margin] / moving_count,
        "selected_transport_block_fraction": selection[best_margin] / block_count,
        "base_flow_l1_pixels_128": flow_base / flow_count,
        "refined_flow_l1_pixels_128": flow_refined / flow_count,
    }


def atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--flow-targets", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--evaluation-samples", type=int, default=0, help="0 evaluates every selected training sample.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--init-head", help="Optional v9.1 checkpoint used to initialize only the new head.")
    parser.add_argument("--flow-loss-weight", type=float, default=0.8)
    parser.add_argument("--classification-loss-weight", type=float, default=0.02)
    parser.add_argument("--advantage-loss-weight", type=float, default=0.08)
    parser.add_argument("--expected-loss-weight", type=float, default=0.25)
    parser.add_argument("--candidate-softmin-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--router-only", action="store_true",
        help="Freeze transport features/flows and update only output channels 10:16 used by the router.",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    motion = []
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            target = window["target_frames"].astype(np.float32)
            previous = np.concatenate((window["context_frames"][-1:].astype(np.float32), target[:-1]), axis=0)
        motion.append((float(np.abs(target - previous).mean() / 255.0), index))
    selected = [index for _, index in sorted(motion, reverse=True)[: args.samples]]
    dataset = PilotDataset(Path(args.windows), Path(args.flow_targets), parent, names, selected)
    if args.evaluation_samples and args.evaluation_samples < len(selected):
        positions = np.linspace(0, len(selected) - 1, args.evaluation_samples, dtype=np.int64)
        evaluation_selected = [selected[int(position)] for position in positions]
    else:
        evaluation_selected = selected
    evaluation_dataset = PilotDataset(Path(args.windows), Path(args.flow_targets), parent, names, evaluation_selected)
    active_mean, active_std = active_statistics(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    checkpoint = Path(args.flow_checkpoint)
    with np.load(checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        base_model = MultiSourceActionFlowUNet(int(config["base_channels"]))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(state["state_dict"], strict=True)
    base_model = base_model.to(device).eval()
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        flow_mean = torch.from_numpy(normalization["mean"]).to(device)
        flow_std = torch.from_numpy(normalization["std"]).to(device)
    active_mean, active_std = active_mean.to(device), active_std.to(device)
    model = ProtectedLayeredFlowV91(base_model, args.base_channels).to(device)
    if args.init_head:
        initialization = torch.load(args.init_head, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(initialization["state_dict"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("base_flow_model.")]
        if unexpected or missing:
            raise ValueError(f"invalid v9.1 head initialization: missing={missing} unexpected={unexpected}")
        print(json.dumps({"event": "loaded_head", "path": str(Path(args.init_head).resolve()), "step": initialization.get("step")}), flush=True)
    if args.router_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.output.weight.requires_grad_(True)
        model.output.bias.requires_grad_(True)
        weight_mask = torch.zeros_like(model.output.weight)
        bias_mask = torch.zeros_like(model.output.bias)
        weight_mask[10:16] = 1
        bias_mask[10:16] = 1
        model.output.weight.register_hook(lambda gradient: gradient * weight_mask)
        model.output.bias.register_hook(lambda gradient: gradient * bias_mask)
        print(json.dumps({"event": "router_only", "trainable_output_channels": [10, 16]}), flush=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.0 if args.router_only else 1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    selected_records = [{"window": names[index], "motion_score": score} for score, index in sorted(motion, reverse=True)[: args.samples]]
    history = []
    loss_weights = {
        "texture": 0.15, "edge": 0.10, "flow": args.flow_loss_weight, "visibility": 0.04,
        "classification": args.classification_loss_weight, "advantage": args.advantage_loss_weight,
        "expected": args.expected_loss_weight, "candidate_softmin": args.candidate_softmin_loss_weight,
        "residual": 0.002,
    }
    initial = evaluate(evaluation_loader, model, device, flow_mean, flow_std, active_mean, active_std)
    history.append({"step": 0, **initial})
    best = initial["router_rgb_mae"]
    best_oracle = float("inf")
    print(json.dumps(history[-1]), flush=True)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        model.train(); model.base_flow_model.eval()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        parent_batch, context, history_actions, future, target, teacher, _ = batch
        parent_batch, context, target = (frames(value, device) for value in (parent_batch, context, target))
        actions = torch.cat((history_actions, future), dim=1).to(device).float()
        teacher = teacher.to(device, non_blocking=True).float()
        active, arm = model.active_arm_actions(actions)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(context, (actions - flow_mean) / flow_std, (active - active_mean) / active_std, arm, parent_batch)
            loss, parts = training_loss(result, context, target, teacher, model.base_flow_model, loss_weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss), **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(evaluation_loader, model, device, flow_mean, flow_std, active_mean, active_std)
            history.append({"step": step, **metrics}); print(json.dumps(history[-1]), flush=True)
            checkpoint_value = {
                "format": "track2-protected-layered-flow-v9.1", "state_dict": {key: value for key, value in model.state_dict().items() if not key.startswith("base_flow_model.")},
                "step": step, "metrics": metrics, "active_mean": active_mean.cpu(), "active_std": active_std.cpu(),
                "base_channels": args.base_channels, "max_residual_flow": model.max_residual_flow,
            }
            atomic_save(checkpoint_value, output / "latest.pt")
            if metrics["router_rgb_mae"] < best:
                best = metrics["router_rgb_mae"]; atomic_save(checkpoint_value, output / "best.pt")
            if metrics["block4_oracle_rgb_mae"] < best_oracle:
                best_oracle = metrics["block4_oracle_rgb_mae"]
                atomic_save(checkpoint_value, output / "best_flow.pt")
            model.train(); model.base_flow_model.eval()
    gate = history[-1]
    manifest = {
        "format": "track2-protected-layered-flow-v9.1-pilot", "samples": selected_records,
        "flow_checkpoint": str(checkpoint.resolve()), "flow_targets": str(Path(args.flow_targets).resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "router_only": args.router_only,
        "evaluation_sample_count": len(evaluation_dataset),
        "init_head": str(Path(args.init_head).resolve()) if args.init_head else None, "loss_weights": loss_weights,
        "history": history, "hard_gate": {
            "candidate_oracle_improvement_required_percent": 35.0, "router_improvement_required_percent": 5.0,
            "candidate_oracle_pass": gate["block4_oracle_improvement_percent"] >= 35.0,
            "router_pass": gate["router_improvement_percent"] >= 5.0,
            "passed": gate["block4_oracle_improvement_percent"] >= 35.0 and gate["router_improvement_percent"] >= 5.0,
        },
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
