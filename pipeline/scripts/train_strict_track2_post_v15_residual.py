#!/usr/bin/env python3
"""Train a bounded residual after frozen complete-V15 predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_reward_aligned_autoregressive_unet import official_prompts, reward_progress_loss
from wam_pipeline.post_v15_residual import PostV15ResidualUNet, parameter_count


class CachedSequences(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.records = self.manifest["records"]
        self.paths = [root / record["path"] for record in self.records]
        self.arm_right = np.asarray([record["arm"] == "right" for record in self.records])
        self.is_synthetic = np.asarray([record["source"] == "synthetic" for record in self.records])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as values:
            return (
                torch.from_numpy(values["context_last"].copy()),
                torch.from_numpy(values["future_actions"].astype(np.float32).copy()),
                torch.from_numpy(values["baseline"].copy()),
                torch.from_numpy(values["target"].copy()),
                torch.tensor(bool(values["arm_right"])),
                str(values["instruction"]),
            )


def active_actions(future: torch.Tensor, arm_right: torch.Tensor) -> torch.Tensor:
    left, right = future[:, :, :7], future[:, :, 7:]
    return torch.where(arm_right[:, None, None], right, left)


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255)


def rollout(model, baseline, context_last, actions, arm_right, mean, std):
    batch, time = baseline.shape[:2]
    normalized = (actions - mean) / std
    context = context_last[:, None].expand(-1, time, -1, -1, -1)
    horizons = torch.arange(1, time + 1, device=baseline.device, dtype=baseline.dtype) / time
    output, detail = model(
        baseline.flatten(0, 1),
        context.flatten(0, 1),
        normalized.flatten(0, 1),
        horizons[None].expand(batch, -1).reshape(-1),
        arm_right[:, None].expand(-1, time).reshape(-1),
    )
    return output.unflatten(0, (batch, time)), {
        key: value.unflatten(0, (batch, time)) for key, value in detail.items()
    }


def visual_loss(output, baseline, target, context_last, detail, protect_weight: float):
    previous_target = torch.cat((context_last[:, None], target[:, :-1]), 1)
    previous_output = torch.cat((context_last[:, None], output[:, :-1]), 1)
    motion = (target - previous_target).abs().mean(2, keepdim=True)
    weight = 1 + 2 * F.max_pool2d(
        (motion > 3 / 255).flatten(0, 1).float(), 9, 1, 4
    ).unflatten(0, motion.shape[:2])
    pixel = ((output - target).abs() * weight).sum() / (weight.sum() * 3)
    temporal = (
        ((output - previous_output) - (target - previous_target)).abs() * weight
    ).sum() / (weight.sum() * 3)
    flat_output, flat_target = output.flatten(0, 1), target.flatten(0, 1)
    output_high = flat_output - F.avg_pool2d(flat_output, 3, 1, 1, count_include_pad=False)
    target_high = flat_target - F.avg_pool2d(flat_target, 3, 1, 1, count_include_pad=False)
    texture = (output_high - target_high).abs().mean()
    accurate = (baseline - target).abs().mean(2, keepdim=True) <= (2 / 255)
    protect = (
        (output - baseline).abs()[accurate.expand_as(output)].mean()
        if accurate.any() else output.new_zeros(())
    )
    magnitude = detail["residual"].abs().mean()
    gate = detail["gate"].mean()
    loss = pixel + 0.35 * temporal + 0.15 * texture + protect_weight * protect + 0.01 * magnitude
    return loss, {
        "pixel": pixel,
        "temporal": temporal,
        "texture": texture,
        "protect": protect,
        "residual": magnitude,
        "gate": gate,
    }


def save(path: Path, model, mean, std, metadata):
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "post_v15_residual.pt.tmp"
    torch.save({
        "format": "strict-track2-post-v15-residual-v1",
        "state_dict": model.state_dict(),
        "config": {
            "base_channels": model.base_channels,
            "maximum_residual_255": model.maximum_residual * 255,
        },
        "active_action_mean": mean.cpu(),
        "active_action_std": std.cpu(),
    }, temporary)
    os.replace(temporary, path / "post_v15_residual.pt")
    (path / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--reset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reward-loss-weight", type=float, default=0.02)
    parser.add_argument("--maximum-residual-255", type=float, default=8.0)
    parser.add_argument("--initial-gate-probability", type=float, default=0.05)
    parser.add_argument("--protect-weight", type=float, default=5.0)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.checkpoint_interval, args.batch_size, args.base_channels) < 1:
        raise SystemExit("counts must be positive")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = CachedSequences(Path(args.cache))
    weights = np.zeros(len(dataset), dtype=np.float64)
    for synthetic in (False, True):
        for right in (False, True):
            mask = (dataset.is_synthetic == synthetic) & (dataset.arm_right == right)
            if not mask.any():
                raise ValueError(f"empty source/arm stratum: synthetic={synthetic} right={right}")
            weights[mask] = 0.25 / mask.sum()
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    iterator = iter(loader)
    all_actions = []
    for path, right in zip(dataset.paths, dataset.arm_right):
        with np.load(path, allow_pickle=False) as values:
            future = values["future_actions"]
        all_actions.append(future[:, 7:] if right else future[:, :7])
    action_values = np.concatenate(all_actions).astype(np.float32)
    device = torch.device(args.device)
    mean = torch.from_numpy(action_values.mean(0)).to(device)
    std = torch.from_numpy(action_values.std(0).clip(1e-6)).to(device)
    model = PostV15ResidualUNet(
        args.base_channels, args.maximum_residual_255, args.initial_gate_probability
    ).to(device)
    initialization = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        initialization = torch.load(
            init_path / "post_v15_residual.pt", map_location="cpu", weights_only=True
        )
        if initialization.get("format") != "strict-track2-post-v15-residual-v1":
            raise ValueError("unsupported post-V15 residual initialization")
        config = initialization["config"]
        if int(config["base_channels"]) != args.base_channels:
            raise ValueError("initialization base channel mismatch")
        if abs(float(config["maximum_residual_255"]) - args.maximum_residual_255) > 1e-6:
            raise ValueError("initialization residual bound mismatch")
        model.load_state_dict(initialization["state_dict"], strict=True)
        mean = initialization["active_action_mean"].to(device)
        std = initialization["active_action_std"].to(device)
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        args.reward_checkpoint, config={"t5_model_name": args.t5_model}
    ).to(device).eval().requires_grad_(False)
    left_prompts = official_prompts(Path(args.reset_manifest), "left", 4)
    right_prompts = official_prompts(Path(args.reset_manifest), "right", 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output_root = Path(args.output)
    history = []
    for step in range(1, args.steps + 1):
        try:
            raw_context, raw_future, raw_baseline, raw_target, arm_right, instruction = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_context, raw_future, raw_baseline, raw_target, arm_right, instruction = next(iterator)
        context = frames(raw_context[:, None], device)[:, 0]
        baseline = frames(raw_baseline, device)
        target = frames(raw_target, device)
        future = raw_future.to(device)
        arm_right = arm_right.to(device).bool()
        actions = active_actions(future, arm_right)
        prompts = []
        for index, (right, exact) in enumerate(zip(arm_right.tolist(), instruction)):
            choices = right_prompts if right else left_prompts
            prompts.append(exact or choices[(step + index) % len(choices)])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, detail = rollout(model, baseline, context, actions, arm_right, mean, std)
            image_loss, parts = visual_loss(
                prediction, baseline, target, context, detail, args.protect_weight
            )
        with torch.autocast(device_type=device.type, enabled=False):
            task_loss, reward_parts = reward_progress_loss(
                reward_model, prediction.float(), target.float(), context.float(), prompts,
                "probability", 1e-5,
            )
        loss = image_loss.float() + args.reward_loss_weight * task_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "visual_loss": float(image_loss.detach().cpu()),
                "reward_loss": float(task_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in parts.items()},
                **{key: float(value.cpu()) for key, value in reward_parts.items()},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps:
            metadata = {
                "format": "strict-track2-post-v15-residual-training-v1",
                "checkpoint_step": step,
                "cache": str(Path(args.cache).resolve()),
                "training": vars(args),
                "train_sequences": len(dataset),
                "parameters": parameter_count(model),
                "initialization_checkpoint": (
                    str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
                ),
                "sampling": {
                    "official_fraction": 0.5,
                    "synthetic_fraction": 0.5,
                    "left_fraction": 0.5,
                    "right_fraction": 0.5,
                },
                "prompt_protocol": {"left": left_prompts, "right": right_prompts},
                "frozen_reward_model_modified": False,
                "history": history,
            }
            save(output_root / f"checkpoint_step_{step:06d}", model, mean, std, metadata)


if __name__ == "__main__":
    main()
